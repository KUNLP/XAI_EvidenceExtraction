import os
import torch
import timeit
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src.functions.utils import load_examples, set_seed, to_list, load_input_data
from src.functions.processor_sent import SquadResult
from src.functions.evaluate_v1_0 import eval_during_train, f1_score
from src.functions.squad_metric import (
    compute_predictions_logits, restore_prediction, restore_prediction_2
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
            try:
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
                total_loss, hop_1_loss, hop_2_loss, hop_3_loss, _, _, sampled_evidence_sentence = outputs[:7]
                evidence_path = to_list(sampled_evidence_sentence)
                cur_example = examples[step]
                cur_feature = features[step]

                refine_input_ids = []
                refine_attention_masks = []
                eval_features = []
                for sample in evidence_path:
                    eval_feature = []

                    hop_1_ex_id = sample[0] // args.max_sent_num
                    hop_1_sent_num = (sample[0] % args.max_sent_num) - 1
                    hop_1_evidence_sentence = cur_example[cur_feature[hop_1_ex_id].example_id].doc_sent_tokens[hop_1_sent_num]

                    hop_2_ex_id = sample[1] // args.max_sent_num
                    hop_2_sent_num = (sample[1] % args.max_sent_num) - 1

                    hop_2_evidence_sentence = cur_example[cur_feature[hop_2_ex_id].example_id].doc_sent_tokens[hop_2_sent_num]


                    hop_3_ex_id = sample[2] // args.max_sent_num
                    hop_3_sent_num = (sample[2] % args.max_sent_num) - 1
                    hop_3_evidence_sentence = cur_example[cur_feature[hop_3_ex_id].example_id].doc_sent_tokens[hop_3_sent_num]

                    query = [tokenizer.cls_token_id] + cur_feature[0].truncated_query

                    refine_context = ['[SEP]'] + hop_1_evidence_sentence + hop_2_evidence_sentence + hop_3_evidence_sentence
                    eval_feature.append(query + refine_context)
                    eval_features.append(eval_feature)
                    context_token_ids = tokenizer.convert_tokens_to_ids(refine_context)
                    refine_input_id = query + context_token_ids
                    refine_input_id = refine_input_id[:args.max_seq_length-1] + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (args.max_seq_length-len(refine_input_id)-1)
                    refine_attention_mask = [1]*args.max_seq_length if 0 not in refine_input_id else [1]*(refine_input_id.index(0)) + [0]*(args.max_seq_length-refine_input_id.index(0))

                    refine_input_ids.append(refine_input_id)
                    refine_attention_masks.append(refine_attention_mask)
                refine_input_ids = torch.tensor(refine_input_ids, dtype=torch.long).cuda()
                refine_attention_masks = torch.tensor(refine_attention_masks, dtype=torch.long).cuda()
                model.eval()
                start_logit, end_logit = model(input_ids = refine_input_ids, attention_mask=refine_attention_masks)
                model.train()
                batch_size = start_logit.size(0)

                results = []
                for i in range(batch_size):
                    output = [to_list(output[i]) for output in [start_logit, end_logit]]
                    start, end = output
                    # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
                    result = SquadResult(0, start, end, evidence=evidence_path[i])
                    results.append(result)
                preds, scores = restore_prediction_2(results, eval_features, args.n_best_size, tokenizer)
                hop_1_sent_score = []
                hop_2_sent_score = []
                hop_3_sent_score = []


                for idx, pred in enumerate(preds):
                    f1 = f1_score(pred , cur_example[0].answer_text)

                    # if f1 < 0.5:
                    #     f1 = 0.0
                    hop_1_sent_score.append(f1)
                    hop_2_sent_score.append(f1)
                    hop_3_sent_score.append(f1)
                hop_1_weight = torch.tensor(hop_1_sent_score, dtype=torch.float).cuda()
                hop_2_weight = torch.tensor(hop_2_sent_score, dtype=torch.float).cuda()
                hop_3_weight = torch.tensor(hop_3_sent_score, dtype=torch.float).cuda()
                # hop_1_weight = hop_1_weight.view(-1, args.num_samples)
                # hop_2_weight = hop_2_weight.view(-1, args.num_samples*args.num_samples)

                hop_1_loss = hop_1_loss * hop_1_weight
                hop_2_loss = hop_2_loss * hop_2_weight
                hop_3_loss = hop_3_loss * hop_3_weight

                # hop_1_loss = torch.sum(hop_1_loss, -1)
                # hop_2_loss = torch.sum(hop_2_loss, -1)
                hop_loss = torch.sum(hop_1_loss)/hop_1_loss.size(0) + torch.sum(hop_2_loss)/hop_2_loss.size(0) + torch.sum(hop_3_loss)/hop_3_loss.size(0)
                if hop_loss.item() != 0:
                    total_loss = total_loss + hop_loss

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
                        output_dir = os.path.join(args.save_dir, "checkpoint-{}".format(global_step))
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
            except:
                print("Current Step {} Error!".format(global_step))
                continue

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
    output_prediction_file = os.path.join(args.save_dir, "predictions_{}.json".format(global_step))
    output_nbest_file = os.path.join(args.save_dir, "nbest_predictions_{}.json".format(global_step))

    # Yes/No Question을 다룰 경우, 각 정답이 유효할 확률 저장을 위한 파일 생성
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.save_dir, "null_odds_{}.json".format(global_step))
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

    output_dir = os.path.join(args.save_dir, 'eval')
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
        for key in sorted(official_eval_results.keys()):
            logger.info("  %s = %s", key, str(official_eval_results[key]))
            f.write(" {} = {}\n".format(key, str(official_eval_results[key])))

