import os
import json
import re
import string
import pandas as pd
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from rouge_score import rouge_scorer
from transformers import RobertaTokenizerFast

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

class Answer(BaseModel):
    """Question answer"""
    answer: str = Field(description="only one word from the possible answers containing your answer.")
    reason: str = Field(description="reason why did you answer this way.")

disrpt_prompt_template = PromptTemplate.from_template("Define a connection between two sentences. Possible answers are those:\n {options}.\nSentence 1: {sent_1}\nSentence 2: {sent_2}\nGive only one answer from the options and reason why did you choose that. Use JSON for output consist of two fields: `answer` and `reason`.")

tape_prompt_template = PromptTemplate.from_template("Define a connection between two sentences. Possible answers are those:\n {options}.\nSentence 1: {sent_1}\nSentence 2: {sent_2}\nGive only one answer from the options and reason why did you choose that.")



def score_ellipsis(model_name, temperature):
    """Run scoring on the ellipsis corpus dataset."""
    # Load data
    data = pd.read_csv('../data/ellipsis corpus.csv')
    elipsis = [re.sub(r' _', '', i) for i in data['sentence']]
    elipsis = [re.sub(r'_', '', i) for i in elipsis]
    elipsis = [re.sub(r'\n', ' ', i) for i in elipsis]
    answers_golden = data['suggested ellipsis resolution']

    # Initialize model
    model = ChatOpenAI(model=model_name, api_key=API_KEY, base_url=BASE_URL, temperature=temperature)

    # Generate model answers
    start, with_, final = [], [], []
    for text in tqdm(elipsis):
        ans = model.invoke(f'''Дано предложение {text}. Оно содержит эллипсис, в нем пропущена часть информации.
        Постарайся восполнить как можно больше информации, не придумывай и не добавляй того, чего нет в контексте.
        Определи, 1) в каком месте пропущена информация, обозначь это место нижним подчеркиванием. 2) Восполни информацию и
        3) напиши новое предложение с восполненой информацией.
        Ответ дай в формате: изначальное - ответ на 1, эллипсис - ответ на 2, полное - ответ на 3. Ответ должен быть в формате json.''').content

        parsed_ans = json.loads(ans.split('json')[1].strip('```'))
        start.append(parsed_ans['изначальное'])
        with_.append(parsed_ans['эллипсис'])
        final.append(parsed_ans['полное'])

    # Create DataFrame
    model_answers_dict = {'initial': start, 'ellipsis': with_, 'final': final}
    model_answers_df = pd.DataFrame.from_dict(model_answers_dict)
    tokenizer = RobertaTokenizerFast.from_pretrained('blinoff/roberta-base-russian-v0', max_len=512)

    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=tokenizer)

    # Save answers to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ellipsis_corpus_{model_name}_temp{temperature}_{timestamp}.csv"
    model_answers_df.to_csv(filename, index=False)

    # Calculate metrics
    model_answers_clean = [re.sub(r'ё', r'е', ans.translate(str.maketrans('', '', string.punctuation)).lower()) for ans in model_answers_df['ellipsis']]
    golden_answers_clean = [re.sub(r'ё', r'е', ans.translate(str.maketrans('', '', string.punctuation)).lower()) for ans in answers_golden]

    for_metrics = [1 if model_answers_clean[i] == golden_answers_clean[i] else 0 for i in range(len(model_answers_clean))]
    
    # Initialize Rouge Scorer
    tokenizer = RobertaTokenizerFast.from_pretrained('blinoff/roberta-base-russian-v0', max_len=512)
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=tokenizer)

    rouge1_precision = []
    rouge1_recall = []
    rouge1_f = []
    rouge2_precision = []
    rouge2_recall = []
    rouge2_f = []
    rougeL_precision = []
    rougeL_recall = []
    rougeL_f = []

    for sentence_pair in range(len(model_answers_clean)):
        sent_pair_r_score = r_scorer.score(model_answers_clean[sentence_pair], golden_answers_clean[sentence_pair])
        rouge1_precision.append(sent_pair_r_score['rouge1'][0])
        rouge1_recall.append(sent_pair_r_score['rouge1'][1])
        rouge1_f.append(sent_pair_r_score['rouge1'][2])
        rouge2_precision.append(sent_pair_r_score['rouge2'][0])
        rouge2_recall.append(sent_pair_r_score['rouge2'][1])
        rouge2_f.append(sent_pair_r_score['rouge2'][2])
        rougeL_precision.append(sent_pair_r_score['rougeL'][0])
        rougeL_recall.append(sent_pair_r_score['rougeL'][1])
        rougeL_f.append(sent_pair_r_score['rougeL'][2])

    metrics = {
        'accuracy': accuracy_score([1] * len(for_metrics), for_metrics),
        'recall': recall_score([1] * len(for_metrics), for_metrics),
        'f1': f1_score([1] * len(for_metrics), for_metrics),
        'precision': precision_score([1] * len(for_metrics), for_metrics),
        'rouge1_precision': np.mean(rouge1_precision),
        'rouge1_recall': np.mean(rouge1_recall),
        'rouge1_f': np.mean(rouge1_f),
        'rouge2_precision': np.mean(rouge2_precision),
        'rouge2_recall': np.mean(rouge2_recall),
        'rouge2_f': np.mean(rouge2_f),
        'rougeL_precision': np.mean(rougeL_precision),
        'rougeL_recall': np.mean(rougeL_recall),
        'rougeL_f': np.mean(rougeL_f)
    }

    return metrics, filename

def score_disrpt(model_name, temperature):
    """Run scoring on the disrpt.json dataset."""
    # Load data
    with open('../data/disrpt.json', 'r') as f:
        data = json.load(f)

    # Initialize model
    model = ChatOpenAI(model=model_name, api_key=API_KEY, base_url=BASE_URL, temperature=temperature)
    structured_llm = model.with_structured_output(Answer)

    # Generate model answers
    results = []
    for i in tqdm([x for x in data.values() if x.get('sent_2')]):
        try:
            res = structured_llm.invoke(disrpt_prompt_template.invoke({'options': i['choices'],
                                    'sent_1': i['sent_1'],
                                    'sent_2': i['sent_2']}))
            results.append((
                i, res, i['label']
            ))
        except Exception as e:
            print(e)

    # Calculate metrics
    y_true = [i[2].lower() for i in results]
    y_pred = [i[1]['answer'].lower() for i in results]

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro')
    }

    # Save answers to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"disrpt_json_{model_name}_temp{temperature}_{timestamp}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)

    return metrics, filename

def run_all(model_name, temperature):
    """Run all scoring functions and save metrics to JSON."""
    ellipsis_metrics, ellipsis_csv = score_ellipsis(model_name, temperature)
    disrpt_metrics, disrpt_csv = score_disrpt(model_name, temperature)

    # Combine metrics
    all_metrics = {
        'ellipsis_corpus': ellipsis_metrics,
        'disrpt_json': disrpt_metrics
    }

    # Save metrics to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"metrics_{model_name}_temp{temperature}_{timestamp}.json"
    with open(metrics_filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    return all_metrics, [ellipsis_csv, disrpt_csv, metrics_filename]

# Example usage
if __name__ == "__main__":
    model_name = "gpt-4o-mini"
    temperature = 0
    run_all(model_name, temperature)
