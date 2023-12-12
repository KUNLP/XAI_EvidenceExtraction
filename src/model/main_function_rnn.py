from torch.nn import functional as F
import os
import torch
import timeit
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src.functions.utils import load_examples, set_seed, to_list, load_input_data
from src.functions.processor_sent import SquadResult
from src.functions.evaluate_v1_0 import eval_during_train, f1_score
from src.functions.hotpotqa_metric import eval
from src.functions.squad_metric import (
    compute_predictions_logits, restore_prediction, restore_prediction2
)


def train(args, model, tokenizer, logger):
    # 학습에 사용하기 위한 dataset Load
    examples, features = load_examples(args, tokenizer, evaluate=False, output_examples=True)

    # optimization 최적화 schedule 을 위한 전체 training step 계산
    t_total = len(features) // args.gradient_accumulation_steps * args.num_train_epochs

    # Layer에 따른 가중치 decay 적용
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # optimizer 및 scheduler 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(features))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1

    tr_loss, logging_loss = 0.0, 0.0

    # loss buffer 초기화
    model.zero_grad()

    set_seed(args)

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(features):
            # if not args.from_init_weight:
            #     if global_step< int(args.checkpoint):
            #         global_step+=1
            #         continue
            # try:
            model.train()
            all_input_ids = torch.tensor([feature.input_ids for feature in batch], dtype=torch.long).cuda()
            all_attention_masks = torch.tensor([feature.attention_mask for feature in batch], dtype=torch.long).cuda()
            all_token_type_ids = torch.tensor([feature.token_type_ids for feature in batch], dtype=torch.long).cuda()
            all_sent_masks = torch.tensor([feature.sent_mask for feature in batch], dtype=torch.long).cuda()
            all_start_positions = torch.tensor([feature.start_position for feature in batch], dtype=torch.long).cuda()
            all_end_positions = torch.tensor([feature.end_position for feature in batch], dtype=torch.long).cuda()
            all_sent_label = torch.tensor([feature.sent_label for feature in batch], dtype=torch.long).cuda()
            if torch.sum(all_start_positions).item() == 0:
                continue
            # 모델에 입력할 입력 tensor 저장
            inputs = {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
                "token_type_ids": all_token_type_ids,
                "sent_masks": all_sent_masks,
                "start_positions": all_start_positions,
                "end_positions": all_end_positions,

            }

            # Loss 계산 및 저장
            outputs = model(**inputs)
            total_loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps

            total_loss.backward()
            tr_loss += total_loss.item()

            # Loss 출력
            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step+1),total_loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # model save
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 모델 저장 디렉토리 생성
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # 학습된 가중치 및 vocab 저장
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Validation Test!!
                    logger.info("***** Eval results *****")
                    evaluate(args, model, tokenizer, logger, global_step=global_step)
            # except:
            #     print("Current Step {} Error!".format(global_step))
            #     continue

    return global_step, tr_loss / global_step

def sample_train(args, model, tokenizer, logger):
    # 학습에 사용하기 위한 dataset Load
    examples, features = load_examples(args, tokenizer, evaluate=False, output_examples=True)

    # optimization 최적화 schedule 을 위한 전체 training step 계산
    t_total = len(features) // args.gradient_accumulation_steps * args.num_train_epochs

    # Layer에 따른 가중치 decay 적용
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # optimizer 및 scheduler 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(features))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1

    tr_loss, logging_loss = 0.0, 0.0

    # loss buffer 초기화
    model.zero_grad()

    set_seed(args)
    for name, para in model.named_parameters():
        if 'gru' not in name:
            print(name)
            para.requires_grad = False
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(features):
            model.train()
            all_input_ids = torch.tensor([feature.input_ids for feature in batch], dtype=torch.long).cuda()
            all_attention_masks = torch.tensor([feature.attention_mask for feature in batch], dtype=torch.long).cuda()
            all_token_type_ids = torch.tensor([feature.token_type_ids for feature in batch], dtype=torch.long).cuda()
            all_sent_masks = torch.tensor([feature.sent_mask for feature in batch], dtype=torch.long).cuda()
            all_start_positions = torch.tensor([feature.start_position for feature in batch], dtype=torch.long).cuda()
            all_end_positions = torch.tensor([feature.end_position for feature in batch], dtype=torch.long).cuda()
            all_sent_label = torch.tensor([feature.sent_label for feature in batch], dtype=torch.long).cuda()
            if torch.sum(all_start_positions).item() == 0:
                continue
            # 모델에 입력할 입력 tensor 저장
            inputs = {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
                "token_type_ids": all_token_type_ids,
                "sent_masks": all_sent_masks,
                "start_positions": all_start_positions,
                "end_positions": all_end_positions,

            }

            outputs = model(**inputs)
            loss, span_loss, mse_loss, sampled_evidence_scores, start_logits, end_logits, sampled_evidence_sentence = outputs


            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps
            # if loss.item() == 0:
            #     continue
            # loss.backward()

            if args.gradient_accumulation_steps > 1:
                span_loss = span_loss / args.gradient_accumulation_steps
                mse_loss = mse_loss / args.gradient_accumulation_steps
                loss = loss / args.gradient_accumulation_steps
            mse_loss.backward()





            tr_loss += loss.item()

            # Loss 출력
            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step+1),span_loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # model save
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 모델 저장 디렉토리 생성
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # 학습된 가중치 및 vocab 저장
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Validation Test!!
                    logger.info("***** Eval results *****")
                    evaluate(args, model, tokenizer, logger, global_step=global_step)
            # except:
            #     print("Current Step {} Error!".format(global_step))
            #     continue

    return global_step, tr_loss / global_step
def sample_train2(args, model, tokenizer, logger):
    # 학습에 사용하기 위한 dataset Load
    examples, features = load_examples(args, tokenizer, evaluate=False, output_examples=True)

    # optimization 최적화 schedule 을 위한 전체 training step 계산
    t_total = len(features) // args.gradient_accumulation_steps * args.num_train_epochs

    # Layer에 따른 가중치 decay 적용
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # optimizer 및 scheduler 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(features))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1

    tr_loss, logging_loss = 0.0, 0.0

    # loss buffer 초기화
    model.zero_grad()

    set_seed(args)
    # for name, para in model.named_parameters():
    #     if 'gru' not in name:
    #         print(name)
    #         para.requires_grad = False
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(features):
            model.train()
            all_input_ids = torch.tensor([feature.input_ids for feature in batch], dtype=torch.long).cuda()
            all_attention_masks = torch.tensor([feature.attention_mask for feature in batch], dtype=torch.long).cuda()
            all_token_type_ids = torch.tensor([feature.token_type_ids for feature in batch], dtype=torch.long).cuda()
            all_sent_masks = torch.tensor([feature.sent_mask for feature in batch], dtype=torch.long).cuda()
            all_start_positions = torch.tensor([feature.start_position for feature in batch], dtype=torch.long).cuda()
            all_end_positions = torch.tensor([feature.end_position for feature in batch], dtype=torch.long).cuda()
            all_question_type = torch.tensor([batch[0].question_type], dtype=torch.long).cuda()

            if torch.sum(all_start_positions).item() == 0:
                continue
            # 모델에 입력할 입력 tensor 저장
            inputs = {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
                "token_type_ids": all_token_type_ids,
                "sent_masks": all_sent_masks,
                "start_positions": all_start_positions,
                "end_positions": all_end_positions,
                #"question_type": all_question_type

            }

            outputs = model(**inputs)
            loss,  sampled_evidence_scores, mask, start_logits, end_logits, sampled_evidence_sentence = outputs
            predicted_answer = []
            evidence_predicted_answer = []
            # print("\n".join([str(e) for e in sampled_evidence_sentence.tolist()]))
            for path in range(num_samples):
                all_results = []
                start_logit = start_logits[:, :, path]
                end_logit = end_logits[:, :, path]
                batch_size = start_logits.size(0)
                for i in range(batch_size):
                    # feature 고유 id로 접근하여 원본 q_id 저장
                    # 각 feature는 유일한 q_id를 갖고 있지 않음
                    # ==> context가 긴 경우, context를 분할하여 여러 개의 데이터로 변환하기 때문!
                    eval_feature = batch[i]

                    # 입력 질문에 대한 N개의 결과 저장하기위해 q_id 저장
                    unique_id = int(eval_feature.unique_id)

                    # outputs = [start_logits, end_logits]
                    output = [to_list(output[i]) for output in [start_logit, end_logit]]

                    # start_logits: [batch_size, max_length]
                    # end_logits: [batch_size, max_length]
                    start, end = output

                    # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
                    result = SquadResult(unique_id, start, end)

                    # feature에 종속되는 최종 출력 값을 리스트에 저장
                    all_results.append(result)
                prediction = restore_prediction(examples[step], batch, all_results, args.n_best_size, args.do_lower_case,
                                                args.verbose_logging, tokenizer)
                predicted_answer.append(prediction)
                evidence_path = sampled_evidence_sentence[path].tolist()

                question = all_input_ids[0, eval_feature.cur_sent_range[0]].tolist()
                evidence_1_feature_index = evidence_path[0]//args.max_sent_num
                evidence_2_feature_index = evidence_path[1]//args.max_sent_num
                evidence_3_feature_index = evidence_path[2] // args.max_sent_num

                evidence_1_sent_num = evidence_path[0] % args.max_sent_num
                evidence_2_sent_num = evidence_path[1] % args.max_sent_num
                evidence_3_sent_num = evidence_path[2] % args.max_sent_num

                evidence_1_sentence = all_input_ids[
                    evidence_1_feature_index, batch[evidence_1_feature_index].cur_sent_range[evidence_1_sent_num]].tolist()
                evidence_2_sentence = all_input_ids[
                    evidence_2_feature_index, batch[evidence_2_feature_index].cur_sent_range[evidence_2_sent_num]].tolist()
                evidence_3_sentence = all_input_ids[
                    evidence_3_feature_index, batch[evidence_3_feature_index].cur_sent_range[evidence_3_sent_num]].tolist()

                tmp_input_ids = question + evidence_1_sentence + evidence_2_sentence + evidence_3_sentence
                tmp_input_ids = tmp_input_ids[:args.max_seq_length-1] + [tokenizer.sep_token_id]
                tokens = tokenizer.convert_ids_to_tokens(tmp_input_ids)
                tmp_attention_mask = torch.zeros([1, args.max_seq_length], dtype=torch.long)
                input_mask = [e for e in range(len(tmp_input_ids))]
                tmp_attention_mask[:, input_mask] = 1
                tmp_input_ids = tmp_input_ids + [tokenizer.pad_token_id] * (args.max_seq_length-len(tmp_input_ids))

                tmp_sentence_mask = [0]*len(question) + [1]*len(evidence_1_sentence) + [2]*len(evidence_2_sentence) + [3]*len(evidence_3_sentence)
                tmp_sentence_mask = tmp_sentence_mask[:args.max_seq_length] + [0]*(args.max_seq_length-len(tmp_sentence_mask))

                tmp_input_ids = torch.tensor([tmp_input_ids], dtype=torch.long).cuda()
                tmp_attention_mask = torch.tensor(tmp_attention_mask, dtype=torch.long).cuda()
                tmp_sentence_mask = torch.tensor([tmp_sentence_mask], dtype=torch.long).cuda()
                inputs = {
                    "input_ids": tmp_input_ids,
                    "attention_mask": tmp_attention_mask,
                    "sent_masks": tmp_sentence_mask,
                }
                e_start_logits, e_end_logits, e_sampled_evidence_sentence = model(**inputs)

                # start_logits: [batch_size, max_length]
                # end_logits: [batch_size, max_length]
                start = e_start_logits[0]
                end = e_end_logits[0]

                # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
                result = [SquadResult(0, start, end)]
                prediction = restore_prediction2(tokens, result, args.n_best_size, tokenizer)

                evidence_predicted_answer.append(prediction)
                # feature에 종속되는 최종 출력 값을 리스트에 저장
            num_samples = predicted_answer.size(0)
            f1_list = [1e-3 for _ in range(num_samples)]
            g_f1_list = [1e-3 for _ in range(num_samples)]
            gold = examples[step][0].answer_text.lower().split(' ')
            gold_list = []
            for word in gold:
                gold_list += tokenizer.tokenize(word)
            gold_list = tokenizer.convert_tokens_to_string(gold_list).strip().split(' ')

            for path in range(num_samples):
                predicted = predicted_answer[path].lower().split(' ')
                e_predicted = evidence_predicted_answer[path].lower().split(' ')
                f1 = sentence_bleu([predicted], e_predicted, weights=(1.0, 0, 0, 0))
                g_f1 = sentence_bleu([gold_list], e_predicted, weights=(1.0, 0, 0, 0))

                f1_list[path] += f1
                g_f1_list[path] += g_f1
            f1_list = torch.tensor(f1_list, dtype=torch.float).cuda()
            g_f1_list = torch.tensor(g_f1_list, dtype=torch.float).cuda()
            #sampled_evidence_scores (10,3, 1, 400)
            #sampled_evidence_sentence (10, 3)

            sampled_evidence_scores = sampled_evidence_scores.squeeze(2)


            # tmp~ : sample별로 추출된 문장 idx들 [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
            s_sampled_evidence_sentence = torch.zeros([num_samples, args.max_dec_len, sampled_evidence_scores.size(-1)], dtype=torch.long).cuda()
            g_sampled_evidence_sentence = torch.zeros([num_samples, args.max_dec_len, sampled_evidence_scores.size(-1)], dtype=torch.long).cuda()
            for idx in range(num_samples):
                sampled_sampled_evidence_sentence = F.one_hot(sampled_evidence_sentence[idx, :],
                                                      num_classes=sampled_evidence_scores.size(-1)).unsqueeze(0)
                negative_sampled_evidence_sentence = torch.sum(sampled_sampled_evidence_sentence, 1, keepdim=True)
                f1 = f1_list[idx]
                g_f1 = g_f1_list[idx]
                if f1.item() < 0.5:
                    s_sampled_evidence_sentence[idx, :, :] = mask - negative_sampled_evidence_sentence
                    f1_list[idx] = 1 - f1

                else:
                    s_sampled_evidence_sentence[idx, :, :] = sampled_sampled_evidence_sentence
                if g_f1.item() < 0.5:
                    g_sampled_evidence_sentence[idx, :, :] = mask - negative_sampled_evidence_sentence
                    g_f1_list[idx] = 1- g_f1
                else:
                    g_sampled_evidence_sentence[idx, :, :] = sampled_sampled_evidence_sentence
            e_div = torch.sum(s_sampled_evidence_sentence, -1)
            g_div = torch.sum(g_sampled_evidence_sentence, -1)
            # if e_div.item() == 3:
            #     e_div = 1
            # if g_div.item() == 3:
            #     g_div = 1
            evidence_nll = -F.log_softmax(sampled_evidence_scores, -1)
            g_evidence_nll = -F.log_softmax(sampled_evidence_scores, -1)

            evidence_nll = evidence_nll  * s_sampled_evidence_sentence
            g_evidence_nll = g_evidence_nll  * g_sampled_evidence_sentence
            f1_list[1:] = f1_list[1:]*0.25
            evidence_nll = torch.mean(torch.sum(evidence_nll, -1)/e_div, -1)
            evidence_nll = evidence_nll * f1_list
            evidence_nll = torch.mean(evidence_nll)

            g_evidence_nll = torch.mean(torch.sum(g_evidence_nll, -1)/g_div, -1)
            g_evidence_nll = g_evidence_nll * g_f1_list
            g_evidence_nll = torch.mean(g_evidence_nll)

            if evidence_nll.item() != 0 and evidence_nll.item() < 1000:
                loss = loss + 0.1 * evidence_nll
            if g_evidence_nll.item() != 0 and evidence_nll.item() < 1000:
                loss = loss + 0.1 * g_evidence_nll
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()





            tr_loss += loss.item()

            # Loss 출력
            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step+1),loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # model save
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 모델 저장 디렉토리 생성
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # 학습된 가중치 및 vocab 저장
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Validation Test!!
                    logger.info("***** Eval results *****")
                    evaluate(args, model, tokenizer, logger, global_step=global_step)
            # except:
            #     print("Current Step {} Error!".format(global_step))
            #     continue

    return global_step, tr_loss / global_step
# 정답이 사전부착된 데이터로부터 평가하기 위한 함수
def evaluate(args, model, tokenizer, logger, global_step = ""):
    # 데이터셋 Load
    try:
        examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)
    except:
        return None
    # 최종 출력 파일 저장을 위한 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)



    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(features))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 모델 출력을 저장하기위한 리스트 선언
    all_results = []

    # 평가 시간 측정을 위한 time 변수
    start_time = timeit.default_timer()
    model.eval()
    tmp_scores = []
    for batch_idx, batch in enumerate(features):
        # 모델을 평가 모드로 변경

        all_input_ids = torch.tensor([feature.input_ids for feature in batch], dtype=torch.long).cuda()
        all_attention_masks = torch.tensor([feature.attention_mask for feature in batch], dtype=torch.long).cuda()
        all_token_type_ids = torch.tensor([feature.token_type_ids for feature in batch], dtype=torch.long).cuda()
        all_sent_masks = torch.tensor([feature.sent_mask for feature in batch], dtype=torch.long).cuda()

        with torch.no_grad():
            # 평가에 필요한 입력 데이터 저장
            inputs = {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
                "token_type_ids": all_token_type_ids,
                "sent_masks": all_sent_masks,
            }
            # outputs = (start_logits, end_logits)
            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]
            outputs = model(**inputs)


            # 1,000
            # 입력 데이터 별 고유 id 저장 (feature와 dataset에 종속)
            example_indices = batch[-1]
            batch_size = all_input_ids.size(0)
        for i in range(batch_size):
            # feature 고유 id로 접근하여 원본 q_id 저장
            # 각 feature는 유일한 q_id를 갖고 있지 않음
            # ==> context가 긴 경우, context를 분할하여 여러 개의 데이터로 변환하기 때문!
            eval_feature = batch[i]

            # 입력 질문에 대한 N개의 결과 저장하기위해 q_id 저장
            unique_id = int(eval_feature.unique_id)

            # outputs = [start_logits, end_logits]
            output = [to_list(output[i]) for output in outputs]

            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]

            start_logits, end_logits, evidence = output

            # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
            result = SquadResult(unique_id, start_logits, end_logits, evidence)

            # feature에 종속되는 최종 출력 값을 리스트에 저장
            all_results.append(result)
    #     refine_input_ids = []
    #     refine_attention_masks = []
    #     eval_features = []
    #     evidence = outputs[2]
    #     evidence_path = to_list(evidence)[0]
    #     cur_example = examples[batch_idx]
    #     cur_feature = features[batch_idx]
    #
    #     eval_feature = []
    #
    #     hop_1_ex_id = evidence_path[0] // args.max_sent_num
    #     hop_1_sent_num = (evidence_path[0] % args.max_sent_num) - 1
    #     hop_1_evidence_sentence = cur_example[cur_feature[hop_1_ex_id].example_id].doc_sent_tokens[hop_1_sent_num]
    #
    #     hop_2_ex_id = evidence_path[1] // args.max_sent_num
    #     hop_2_sent_num = (evidence_path[1] % args.max_sent_num) - 1
    #
    #     hop_2_evidence_sentence = cur_example[cur_feature[hop_2_ex_id].example_id].doc_sent_tokens[hop_2_sent_num]
    #
    #     hop_3_ex_id = evidence_path[2] // args.max_sent_num
    #     hop_3_sent_num = (evidence_path[2] % args.max_sent_num) - 1
    #     hop_3_evidence_sentence = cur_example[cur_feature[hop_3_ex_id].example_id].doc_sent_tokens[hop_3_sent_num]
    #
    #     query = [tokenizer.cls_token_id] + cur_feature[0].truncated_query
    #
    #     refine_context = ['[SEP]'] + hop_1_evidence_sentence + hop_2_evidence_sentence + hop_3_evidence_sentence
    #     eval_feature.append(query + refine_context)
    #     eval_features.append(eval_feature)
    #     context_token_ids = tokenizer.convert_tokens_to_ids(refine_context)
    #     refine_input_id = query + context_token_ids
    #     refine_input_id = refine_input_id[:args.max_seq_length - 1] + [tokenizer.sep_token_id] + [
    #                                                                                                  tokenizer.pad_token_id] * (
    #                                                                                              args.max_seq_length - len(
    #                                                                                                  refine_input_id) - 1)
    #     refine_attention_mask = [1] * args.max_seq_length if 0 not in refine_input_id else [1] * (
    #     refine_input_id.index(0)) + [0] * (args.max_seq_length - refine_input_id.index(0))
    #
    #     refine_input_ids.append(refine_input_id)
    #     refine_attention_masks.append(refine_attention_mask)
    #     refine_input_ids = torch.tensor(refine_input_ids, dtype=torch.long).cuda()
    #     refine_attention_masks = torch.tensor(refine_attention_masks, dtype=torch.long).cuda()
    #
    #     start_logit, end_logit = model(input_ids=refine_input_ids, attention_mask=refine_attention_masks)
    #     batch_size = start_logit.size(0)
    #
    #     results = []
    #     for i in range(batch_size):
    #         output = [to_list(output[i]) for output in [start_logit, end_logit]]
    #         start, end = output
    #         # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
    #         result = SquadResult(0, start, end, evidence=evidence_path)
    #         results.append(result)
    #     preds, scores = restore_prediction_2(results, eval_features, args.n_best_size, tokenizer)
    #     for idx, pred in enumerate(preds):
    #         f1 = f1_score(pred, cur_example[0].answer_text)
    #         tmp_scores.append(f1)
    # print(len(tmp_scores))
    # print(sum(tmp_scores)/len(tmp_scores))
    # 평가 시간 측정을 위한 time 변수
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(features))

    # 최종 예측 값을 저장하기 위한 파일 생성
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(global_step))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(global_step))

    # Yes/No Question을 다룰 경우, 각 정답이 유효할 확률 저장을 위한 파일 생성
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(global_step))
    else:
        output_null_log_odds_file = None

    # q_id에 대한 N개의 출력 값의 확률로 부터 가장 확률이 높은 최종 예측 값 저장
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_sent_num,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    output_dir = os.path.join(args.output_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # KorQuAD 평가 스크립트 기반 성능 저장을 위한 파일 생성
    output_eval_file = os.path.join(output_dir, "eval_result_{}_{}.txt".format(
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        global_step))

    logger.info("***** Official Eval results *****")
    with open(output_eval_file, "w", encoding='utf-8') as f:
        # KorQuAD 평가 스크립트 기반의 성능 측정
        official_eval_results = eval_during_train(args, global_step)
        # official_eval_results = eval(output_prediction_file, os.path.join(args.data_dir, args.predict_file))
        for key in sorted(official_eval_results.keys()):
            logger.info("  %s = %s", key, str(official_eval_results[key]))
            f.write(" {} = {}\n".format(key, str(official_eval_results[key])))

