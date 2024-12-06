import os
import argparse
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import json
import numpy as np
import torch
from transformers import set_seed
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset import *
from preprocess import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
VLLM_GPU_NUMS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_model = SentenceTransformer('embedding_model/Labse', local_files_only=True).to(device)  # 🔺


# Utils # 
def batch_generator(texts, batch_size):
    # texts: list[str]
    for i in range(0, len(texts), batch_size):
        yield texts[i:i+batch_size]


# check dirs
def check_dir_existence(model_name, dataset):
    dirs = [f'predictions/{model_name}/{dataset}/']
    for dir in dirs:
        if not os.path.exists(dir):
            raise ValueError('Prediction and prompt path not exist, please create them.')


def extract_complete_sentences(text, max_words=30):
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    last_sentence = sentences[-1] if sentences else text
    is_complete = re.match(r'.*[.!?]$', last_sentence) is not None
    output = ""
    if is_complete:
        output = sentences[0]
        words_num = len(output.split(" "))
        if len(sentences) > 1:
            for s in sentences[1:]:
                words_num += len(s.split(" "))
                if words_num > max_words:
                    break
                output += f" {s}"
    else:
        words = text.split(" ")
        output = " ".join(words[:30]) + "."
    return output


def evaluation(gold_samples, predictions, labels):
    """
    gold_samples: list[json_obj] → ({"text":..., "label":[["e","t"],...]})
    predictions: list[json_obj] → ({"entity":"type", ...})
    # Return: json {precision, recall, f1}
    """
    TP, FP, FN = 0, 0, 0
    for gold_obj, preds in zip(gold_samples, predictions):
        text, gold_entities = gold_obj["text"], gold_obj["label"]
        str_golds, str_preds = [], []
        for pair in gold_entities:
            str_golds.append(f'{pair[0]}:{pair[1]}')
        for pred_e, pred_t in preds.items():
            if (pred_e in text) and (pred_t in labels):
                str_preds.append(f'{pred_e}:{pred_t}')
        str_golds, str_preds = list(set(str_golds)), list(set(str_preds))
        shared = list(set(str_golds)&set(str_preds))
        tp = len(shared)
        fp, fn = len(str_preds)-tp, len(str_golds)-tp
        TP, FP, FN = TP+tp, FP+fp, FN+fn
    try:
        p, r = TP/(TP+FP), TP/(TP+FN)
        f1 = 2*p*r/(p+r)
    except:
        p, r, f1 = 0, 0, 0  # some error
    return {"precision":p, "recall":r, "f1": f1, "gold":TP+FN, "pred":TP+FP, "correct": TP}


def repair_string(s):
    s = s.replace('": "', '":"')
    pattern = r'"([^"]+)":"([^"]+)"'
    matches = re.findall(pattern, s)
    tmp = []
    for k,v in matches:
        tmp.append(f'"{k}": "{v}"')
    repaired_s = '{' + f'{", ".join(tmp)}' + '}'
    return repaired_s


def filter_entity_predictions(predictions, texts, labels):
    filtered_predictions = []
    for preds, text in zip(predictions, texts):
        tmp = {}
        for e, t in preds.items():
            if (e in text) and (t in labels):
                tmp[e] = t
        filtered_predictions.append(tmp)
    return filtered_predictions


def count_pred_entities(predictions):
    num = 0
    for preds in predictions:
        num += len(preds)
    return num

# # 


def generate_predictions(prompt, texts, llm, batch_size, pred_path, labels):
    """
    prompt: str
    texts: list[str]
    Return: list[json_obj] → {"entity1":"type1", ...}
    """
    sampling_params = SamplingParams(temperature=0, top_p=0.5, max_tokens=64, stop='}', seed=0)
    # batch generate # # # # # #
    all_predictions = []
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
    for batch in tqdm(batch_generator(texts, batch_size), total=total_batches, desc=f"Generate predictions"):
        context = [str(prompt + f'Text: {text}.\tAnswer: ') for text in batch]
        response = llm.generate(context, sampling_params, use_tqdm=False)
        response = [str(response[i].outputs[0].text.replace("\n","").strip()+'}') for i in range(len(response))]
        predictions = []
        for r in response:
            try:
                predictions.append(json.loads(r))
            except:
                try:
                    predictions.append(json.loads(repair_string(r)))
                except:
                    r_error = '{"Error":"Error"}'
                    predictions.append(json.loads(repair_string(r_error)))
        batch_texts = [text for text in batch]
        predictions = filter_entity_predictions(predictions, batch_texts, labels)
        all_predictions.extend(predictions)
        with open(pred_path, mode='a', encoding='utf-8') as f:
            for preds in predictions:
                f.write(json.dumps(preds)+'\n')
    return all_predictions


def post_process_preds(historical_preds, thresh_num):
    """
    historical_preds: list[json_obj], historical predictions of a single sample
    Return: json_obj   ({"entity1":"type1", ...})
    """
    weighted_preds = {}     # { "entity":{"type1":weight1, "type2":weight2, ...}, ... }
    counts_preds = {}   # {"entity": pred_num, ...}
    for i, preds in enumerate(historical_preds):
        weight = i + 1
        # weight = 1
        for e, t in preds.items():
            # prediction nums of the entity
            if e not in counts_preds:
                counts_preds[e] = 1
            else:
                counts_preds[e] += 1
            # weights of each type of the entity
            if e not in weighted_preds:
                weighted_preds[e] = {t:weight}
            else:
                if t not in weighted_preds[e]:
                    weighted_preds[e][t] = weight
                else:
                    weighted_preds[e][t] += weight
    thresh_w = len(historical_preds)
    # thresh_n = len(historical_preds) // 2
    thresh_n = thresh_num
    intergrated_pred = {}
    for e in weighted_preds.keys():
        weighted_types = weighted_preds[e]
        max_weight_t = max(weighted_types, key=weighted_types.get)
        if (weighted_types[max_weight_t] > thresh_w) and (counts_preds[e] > thresh_n):
        # if weighted_types[max_weight_t] >  thresh_n:
            intergrated_pred[e] = max_weight_t
    return intergrated_pred


def post_process_predictions(args, historical_predictions, historical_pred_nums):
    """
    historical_predictions: list[list[json_obj]]    ({"entity1":"type1", ...})
    historical_pred_nums: list[int]     pred_entity_nums of each historical step
    # Return: list[json_obj] → {"entity1":"type1", ...}
    """
    if len(historical_predictions) == 1:
        return historical_predictions[0]
    else:
        current_step_num, historical_avg_num = historical_pred_nums[-1], sum(historical_pred_nums[:-1])/(len(historical_pred_nums)-1)
        thresh = len(historical_predictions) // 2
        t = 0.1
        if (args.dataset == 'ACE05-E') or (args.dataset == 'CONLL'):
            t = 0.15
        if current_step_num >= historical_avg_num:
            thresh += ((current_step_num - historical_avg_num) / historical_avg_num) // t
            thresh = min(thresh, len(historical_predictions)-1)     # 约束
        else:
            thresh -= ((historical_avg_num - current_step_num)/historical_avg_num) // t
            thresh = max(thresh, 2)
        intergrated_predictions = []
        for sample_id in range(len(historical_predictions[0])):
            historical_preds = [predictions[sample_id] for predictions in historical_predictions]
            intergrated_pred = post_process_preds(historical_preds, thresh)
            intergrated_predictions.append(intergrated_pred)
        return intergrated_predictions


def split_by_types(predictions, texts, labels):
    """
    # Return: { type: list[json_objs], ...}
    #   json_obj → {'text':..., 'entities':[...], 'text_id':..}
    """
    prediction_splits = {}
    for label in labels:
        subset = []
        for i in range(len(texts)):
            text, text_id, entities = texts[i], i, []
            for pair in predictions[i].items():
                if pair[1] == label:
                    entities.append(pair[0])
            if len(entities) > 0:
                subset.append({"text":text, "entities": entities, "text_id": text_id})
            # save_path = f'tmp_files/tmp_{label}.json'
            # with open(save_path, mode='w', encoding='utf-8') as f:
            #     json.dump(subset, f, ensure_ascii=False, indent=2)
        prediction_splits[label] = subset
    return prediction_splits


# select representative samples # # # #
def get_cluster_center_ids(sentences, embedding_model, n_clusters):
    if len(sentences) <= n_clusters:
        return [i for i in range(len(sentences))]
    embeddings = embedding_model.encode(sentences, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    embeddings_normalized = normalize(embeddings, norm='l2')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_normalized)
    cluster_centers = kmeans.cluster_centers_
    closest_embedding_ids = []
    for center in cluster_centers:
        distances = cdist([center], embeddings_normalized, 'cosine')[0]
        closest_idx = np.argmin(distances)
        closest_embedding_ids.append(closest_idx)
    return closest_embedding_ids

def select_typical_samples(type, num, predictions_type):
    """
    type: str
    num: int
    predictions_type: list[json_objs], refer to split_by_types() → text, entities, text_id)
    # Return: typical_sample_ids_subset, typical_sample_ids: list[int]
    """
    sentences = []
    for obj in predictions_type:
        if len(obj['entities']) == 1:
            s = f'In text "{obj["text"]}", the {type} entity is: {obj["entities"][0]}.'
        else:
            s = f'In text "{obj["text"]}", the {type} entities are: {", ".join(obj["entities"])}.'
        sentences.append(s)
    typical_sample_ids_subset = get_cluster_center_ids(sentences, embedding_model, num)
    typical_sample_ids = []
    for idx in typical_sample_ids_subset:
        typical_sample_ids.append(predictions_type[idx]['text_id'])
    return typical_sample_ids_subset, typical_sample_ids
# # # # # # # # # # # #


# guideline-step1：definitions # # # # # # # # # # # #
def generate_guideline_definition(type, typical_samples, llm, last_step_guide):
    """
    type: str
    typical_samples: list[json_objs] → text, entities, text_id
    last_step_guide: str, 上一step的关于type类型的guideline_definition
    # Return: guideline_definition: str
    """
    if last_step_guide == '':
        prompt = f'The following are some texts which containing the {type} entities. Please summarize the definition of the "{type}" type according to these examples.\n\n'
    else:
        prompt = f'Currently, the "{type}" type is defined as: "{last_step_guide}". '
        prompt = f'The following are some texts which containing the {type} entities. According to these examples, supplement or modify the definition to make it more complete."\n\n'
    for obj in typical_samples:
        if len(obj['entities']) == 1:
            s = f'In text "{obj["text"]}", the {type} entity is: {obj["entities"][0]}.\n'
        else:
            s = f'In text "{obj["text"]}", the {type} entities are: {", ".join(obj["entities"])}.\n'
        prompt += s
    prompt += f'\nAccording to these examples, the "{type}" type refers to: '
    sampling_params = SamplingParams(temperature=0, top_p=0.5, max_tokens=64, seed=0)
    response = llm.generate([str(prompt)], sampling_params, use_tqdm=False)[0].outputs[0].text.replace("\n","").strip()
    guideline_definition = extract_complete_sentences(text=str(response), max_words=30)
    return guideline_definition
# # # # # # # # # # # #


def data_format_convert(predictions_type):
    # [{'text':..., 'entities':..., 'text_id':...}, {...}, ...] → {'entity span':[text_id1, text_id2, ...], ...}
    e_texts = {}
    for obj in predictions_type:
        for entity in obj['entities']:
            if entity in e_texts:
                e_texts[entity].append(obj['text_id'])
            else:
                e_texts[entity] = [obj['text_id']]
    return e_texts

def split_similar(type1, type2, predictions_type1, predictions_type2):
    """
    type: str
    predictions_type: list[json_objs] → text, entities, text_id
    # Return: list[json_obj] → {'entity':..., 'type1':[text_ids], 'type2':[text_ids]}
    """
    e_texts_type1 = data_format_convert(predictions_type1)
    e_texts_type2 = data_format_convert(predictions_type2)
    shared_entities = set(e_texts_type1) & set(e_texts_type2)
    similar_samples = []
    for entity in shared_entities:
        obj = {
            'entity': entity,
            type1: e_texts_type1[entity],
            type2: e_texts_type2[entity]
        }
        similar_samples.append(obj)
    return similar_samples
# # # # # # # # # # # #


# guideline-step2：distinctions # # # # # # # # # # # #
def generate_guideline_diffs(type1, type2, similar_samples, texts, llm):
    prompt = f'The following are some examples of different types of the same entity:\n\n'
    for obj in similar_samples:
        type1_sentences = [texts[i] for i in obj[type1]]
        type2_sentences = [texts[i] for i in obj[type2]]
        s1_ids = get_cluster_center_ids(type1_sentences, embedding_model, 1)
        s2_ids = get_cluster_center_ids(type2_sentences, embedding_model, 1)
        s = f'{obj["entity"]} is a "{type1}" entity in text: "{type1_sentences[s1_ids[0]]}"; and a "{type2}" entity in text: "{type2_sentences[s2_ids[0]]}".\n'
        prompt += s
    # type1
    prompt_1 = prompt + f'\nAccording to these confusing examples, compared to the "{type2}" type, the "{type1}" refers to: '
    sampling_params = SamplingParams(temperature=0, top_p=0.5, max_tokens=64, seed=0)
    response = llm.generate([str(prompt_1)], sampling_params, use_tqdm=False)[0].outputs[0].text.replace("\n","").strip()
    guideline_1 = extract_complete_sentences(text=str(response), max_words=30)
    # type2
    prompt_2 = prompt + f'\nAccording to these confusing examples, compared to the "{type1}" type, the "{type2}" refers to: '
    response = llm.generate([str(prompt_2)], sampling_params, use_tqdm=False)[0].outputs[0].text.replace("\n","").strip()
    guideline_2 = extract_complete_sentences(text=str(response), max_words=30)
    return guideline_1, guideline_2
# # # # # # # # # # # #


def save_prompt(guidelines_definition, guidelines_diffs, few_shot_samples, labels, prompt_path, step):
    s = "'" + "', '".join(labels) + "'"
    prompt = f"Given entity types: [{s}]. The definition of each type are:\n"
    i = 1
    for type, definition, diffs in zip(labels, guidelines_definition, guidelines_diffs):
        prompt += f'({i}) {type}: {definition} {"; ".join(diffs)}\n'
        i += 1
    # v1
    # prompt += "\nPlease recognize the named entities in the given text only belonging to the given types. Provide answer in the following JSON format: {\"entity\": \"type\"}. If there is no entity in the text, return the following empty object: {}.\n"
    # v2
    prompt += "\nPlease recognize the named entities in the given text only belonging to the given types. Provide answer in the following JSON format: {\"entity\": \"type\"}.\n"
    prompt += 'Here are some examples:\n\n'
    for samples in few_shot_samples:
        prompt += f'{samples}\n\n'
    with open(prompt_path, mode='r', encoding='utf-8') as f:
        PROMPTS = json.load(f)
    PROMPTS[f'step{step}'] = str(prompt)
    with open(prompt_path, mode='w', encoding='utf-8') as f:
        PROMPTS = json.dump(PROMPTS, f, ensure_ascii=False, indent=2)
    return prompt


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    parser = argparse.ArgumentParser(description="Args for Auto Guides.")

    parser.add_argument("--dataset", type=str, default='data-debug')
    
    parser.add_argument("--model_name", type=str, default='Llama-3.1-70B-Instruct-GPTQ-INT4', help="The name or path to the model.")
    
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=4, help='The number of prediction iterations.')

    parser.add_argument("--shots_per_type", type=int, default=2, help='The few shot sample number for each type.')

    # for Debug # # # # # # # # # # # # #
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--start_sample_idx", type=int, default=0, help='only valid for the first(start) step')

    args = parser.parse_args()
   
    args.model_path = f'model/{args.model_name}' # 🔺

    set_seed(0)
    check_dir_existence(args.model_name, args.dataset)

    test_data_path = f'dataset/{args.dataset}/test.json'     # 🔺
    label_data_path = f'dataset/{args.dataset}/labels.txt'   # 🔺
    with open(test_data_path, mode='r', encoding='utf-8') as f:
        gold_samples = json.load(f)
    texts = [str(obj['text']) for obj in gold_samples]
    labels = []
    with open(label_data_path, mode='r', encoding='utf-8') as f:
        for line in f:
            labels.append(str(line.strip()))
    print(f'{args.dataset}: {len(texts)} samples.\nLabels: {labels}')

    # Main process
    llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=VLLM_GPU_NUMS, max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=True)
    evaluation_path = f'predictions/{args.model_name}/{args.dataset}/evaluation_results.txt'    # ⭐
    # prompt_path = f'prompts/prompts_{args.dataset}.json'
    prompt_path = f'predictions/{args.model_name}/{args.dataset}/prompts_{args.dataset}.json'   # ⭐
    PROMPT = ''

    # seed guidelines   # ※notice when debug
    if args.dataset == 'CONLL':
        seed_guidelines = ['' for _ in labels]
    else:
        seed_guidelines = ['' for _ in labels]
    if args.dataset == 'CONLL' and args.start_step == 1 and args.model_name == 'demo_model':
        last_step_guides = ['' for _ in labels]
    else:
        last_step_guides = ['' for _ in labels]
    if args.start_step == 0:
        last_step_guides = seed_guidelines
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    for step in range(args.start_step, args.iterations):
        print(f'# Step {step} # # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
        print('----------------------------- generate current step predictions ----------------------------')
        if PROMPT == '':
            with open(prompt_path, mode='r', encoding='utf-8') as f:
                prompts = json.load(f)
            PROMPT = str(prompts[f'step{step}'])
        pred_path = f'predictions/{args.model_name}/{args.dataset}/step{step}.jsonl'
        if step == args.start_step and args.start_sample_idx != 0:  # for debug
            current_predictions = generate_predictions(PROMPT, texts[args.start_sample_idx:], llm, batch_size=args.batch_size, pred_path=pred_path, labels=labels)
            all_predictions = []
            with open(pred_path, mode='r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_predictions.append(json.loads(line))
                    except:
                        import pdb;pdb.set_trace()
            current_predictions = all_predictions
        else:
            current_predictions = generate_predictions(PROMPT, texts, llm, batch_size=args.batch_size, pred_path=pred_path, labels=labels)

        eval_res = evaluation(gold_samples, current_predictions, labels)
        with open(evaluation_path, mode='a', encoding='utf-8') as fe:
            fe.write(f'Step{step}    : precision={eval_res["precision"]},  recall={eval_res["recall"]},  f1={eval_res["f1"]},  gold={eval_res["gold"]}, pred={eval_res["pred"]}, correct={eval_res["correct"]}\n')

        print('----------------------------- post process predictions ----------------------------')
        if step + 1 > 2:
            historical_predictions = []
            for i in range(0, step):
                tmp_path = f'predictions/{args.model_name}/{args.dataset}/step{i}.jsonl'
                tmp_predictions = []
                with open(tmp_path, mode='r', encoding='utf-8') as f:
                    for line in f:
                        tmp_predictions.append(json.loads(line))
                historical_predictions.append(tmp_predictions)
            historical_predictions.append(current_predictions)
            historical_pred_nums = []
            for predictions in historical_predictions:
                pred_nums = count_pred_entities(predictions)
                historical_pred_nums.append(pred_nums)
            processed_predictions = post_process_predictions(args, historical_predictions, historical_pred_nums)

            eval_res = evaluation(gold_samples, processed_predictions, labels)
            with open(evaluation_path, mode='a', encoding='utf-8') as fe:
                fe.write(f'Processed: precision={eval_res["precision"]},  recall={eval_res["recall"]},  f1={eval_res["f1"]},  gold={eval_res["gold"]}, pred={eval_res["pred"]}, correct={eval_res["correct"]}\n')
        else:
            processed_predictions = current_predictions

        print('----------------------------- split predictions by types ----------------------------')
        predictions_split = split_by_types(processed_predictions, texts, labels)

        print('----------------------------- select typical samples ----------------------------')
        typical_num = 8
        typical_sample_ids_subset, typical_sample_ids_whole = [], []
        for label in tqdm(labels, desc='Select typical samples '):
            ids_subset, ids_whole = select_typical_samples(label, typical_num, predictions_split[label])
            typical_sample_ids_subset.append(ids_subset)
            typical_sample_ids_whole.append(ids_whole)
        
        print('----------------------------- generate guidelines-1: definitions ----------------------------')
        guideline_definitions = []
        for t_id, label in tqdm(enumerate(labels), total=len(labels)):
            typical_samples = [predictions_split[label][i] for i in typical_sample_ids_subset[t_id]]
            definition = generate_guideline_definition(label, typical_samples, llm, last_step_guides[t_id])
            guideline_definitions.append(definition)
        last_step_guides = guideline_definitions

        print('----------------------------- generate guidelines-2: differences ----------------------------')
        guideline_differences = [[] for _ in range(len(labels))]
        for i in tqdm(range(len(labels))):
            for j in range(i+1, len(labels)):
                type1, type2 = labels[i], labels[j]
                similar_samples = split_similar(type1, type2, predictions_split[type1], predictions_split[type2])
                if len(similar_samples) > 29:
                    tmp_similar_samples = []
                    for obj in similar_samples:
                        if (len(obj[type1]) > 14) and (len(obj[type2]) > 14):
                            tmp_similar_samples.append(obj)
                    diff1, diff2 = generate_guideline_diffs(type1, type2, tmp_similar_samples[:10], texts, llm)
                    guideline_differences[i].append(diff1)
                    guideline_differences[j].append(diff2)
        
        print('----------------------------- update and save new prompts ----------------------------')
        few_shot_samples = []
        next_step = step + 1
        if next_step > 2:
            num_per_type = args.shots_per_type
            for t_id in range(len(labels)):
                if len(typical_sample_ids_whole[t_id]) > 0:
                    sentences = []
                    for idx in typical_sample_ids_whole[t_id]:
                        sentences.append(texts[idx])
                    center_ids_subset = get_cluster_center_ids(sentences, embedding_model, num_per_type)
                    for idx_subset in center_ids_subset:
                        idx = typical_sample_ids_whole[t_id][idx_subset]    # idx in the whole_set
                        text, answer = texts[idx], json.dumps(processed_predictions[idx])
                        sample = f'Text: {text}\nAnswer: {answer}'
                        few_shot_samples.append(sample)
        else:
            for obj in gold_samples:
                if len(obj["label"]) > 0:
                    text = obj["text"]
                    tmp = {}
                    for pair in obj["label"]:
                        tmp[pair[0]] = pair[1]
                    answer = json.dumps(tmp)
                    break
            sample = f'Text: {text}\nAnswer: {answer}'
            few_shot_samples.append(sample)
        PROMPT = save_prompt(guideline_definitions, guideline_differences, few_shot_samples, labels, prompt_path, next_step)



if __name__ == '__main__':
    pass

    main()
