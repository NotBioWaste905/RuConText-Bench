import os
import json
import re
import string
import random
import pandas as pd
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    classification_report,
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from rouge_score import rouge_scorer
from transformers import RobertaTokenizerFast
from typing import Union, List, Optional
import glob

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")


class Answer(BaseModel):
    """Question answer"""

    answer: str = Field(
        description="only one word from the possible answers containing your answer."
    )
    # reason: str = Field(description="reason why did you answer this way.")


disrpt_prompt_template = PromptTemplate.from_template(
    "Определите связь между двумя предложениями. Возможные следующие варианты ответа:\n {options}.\nПредложение 1: {sent_1}\nПредложение 2: {sent_2}\nДайте только один ответ из предложенных. Используйте JSON для вывода, состоящий из одного поля: `answer`."
)

rudabank_prompt_template = PromptTemplate.from_template(
    "Данное начальное высказывание и ответное высказывание, определите тип ответа из следующих вариантов:\n{options}\n\nНачальное высказывание: {initial_utterance}\nОтветное высказывание: {tagged_utterance}\n\nДайте только один ответ из предложенных. Используйте JSON для вывода, состоящий из одного поля: `answer`."
)


def score_ellipsis(model_name, temperature, n_samples=100):
    """Run scoring on the ellipsis corpus dataset."""
    # Load data
    data = pd.read_csv("data/ellipsis.csv")
    data = data.sample(n=n_samples, random_state=42)

    elipsis = [re.sub(r" _", "", i) for i in data["sentence"]]
    elipsis = [re.sub(r"_", "", i) for i in elipsis]
    elipsis = [re.sub(r"\n", " ", i) for i in elipsis]
    answers_golden = data["suggested ellipsis resolution"]

    # Initialize model
    model = ChatOpenAI(
        model=model_name, api_key=API_KEY, base_url=BASE_URL, temperature=temperature
    )

    # Generate model answers
    start, with_, final = [], [], []
    print("[INFO] Starting ellipsis corpus scoring...")
    for text in tqdm(elipsis, desc="Scoring ellipsis corpus"):
        ans = model.invoke(f"""Дано предложение {text}. Оно содержит эллипсис, в нем пропущена часть информации.
        Постарайся восполнить как можно больше информации, не придумывай и не добавляй того, чего нет в контексте.
        Определи, 1) в каком месте пропущена информация, обозначь это место нижним подчеркиванием. 2) Восполни информацию и
        3) напиши новое предложение с восполненой информацией.
        Ответ дай в формате: изначальное - ответ на 1, эллипсис - ответ на 2, полное - ответ на 3. Ответ должен быть в формате json. В ответе должен быть только JSON в markdown нотации (начинаться с ```json и заканчиваться ```) без дополнительных комментариев.""").content

        try:
            parsed_ans = json.loads(ans.split("json")[1].strip("```"))
            start.append(parsed_ans["изначальное"])
            with_.append(parsed_ans["эллипсис"])
            final.append(parsed_ans["полное"])
        except:
            # print(f"Error parsing answer: {ans}")
            start.append("")
            with_.append("")
            final.append("")
    # Create DataFrame
    model_answers_dict = {"initial": start, "ellipsis": with_, "final": final}
    model_answers_df = pd.DataFrame.from_dict(model_answers_dict)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "blinoff/roberta-base-russian-v0", max_len=512
    )

    r_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], tokenizer=tokenizer
    )

    # Save answers to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ellipsis_{model_name.replace('/','_')}_temp{temperature}_{timestamp}.csv"
    answers_df = pd.DataFrame(
        {
            "task_id": range(len(model_answers_df)),
            "answer": model_answers_df["ellipsis"],
            "correct_answer": answers_golden,
        }
    )
    answers_df.to_csv(filename, index=False)

    # Calculate metrics
    model_answers_clean = [
        re.sub(
            r"ё", r"е", ans.translate(str.maketrans("", "", string.punctuation)).lower()
        )
        for ans in model_answers_df["ellipsis"]
    ]
    golden_answers_clean = [
        re.sub(
            r"ё", r"е", ans.translate(str.maketrans("", "", string.punctuation)).lower()
        )
        for ans in answers_golden
    ]

    for_metrics = [
        1 if model_answers_clean[i] == golden_answers_clean[i] else 0
        for i in range(len(model_answers_clean))
    ]

    # Initialize Rouge Scorer
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "blinoff/roberta-base-russian-v0", max_len=512
    )
    r_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], tokenizer=tokenizer
    )

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
        sent_pair_r_score = r_scorer.score(
            model_answers_clean[sentence_pair], golden_answers_clean[sentence_pair]
        )
        rouge1_precision.append(sent_pair_r_score["rouge1"][0])
        rouge1_recall.append(sent_pair_r_score["rouge1"][1])
        rouge1_f.append(sent_pair_r_score["rouge1"][2])
        rouge2_precision.append(sent_pair_r_score["rouge2"][0])
        rouge2_recall.append(sent_pair_r_score["rouge2"][1])
        rouge2_f.append(sent_pair_r_score["rouge2"][2])
        rougeL_precision.append(sent_pair_r_score["rougeL"][0])
        rougeL_recall.append(sent_pair_r_score["rougeL"][1])
        rougeL_f.append(sent_pair_r_score["rougeL"][2])

    metrics = {
        "accuracy": accuracy_score([1] * len(for_metrics), for_metrics),
        "recall": recall_score([1] * len(for_metrics), for_metrics),
        "f1": f1_score([1] * len(for_metrics), for_metrics),
        "precision": precision_score([1] * len(for_metrics), for_metrics),
        "rouge1_precision": np.mean(rouge1_precision),
        "rouge1_recall": np.mean(rouge1_recall),
        "rouge1_f": np.mean(rouge1_f),
        "rouge2_precision": np.mean(rouge2_precision),
        "rouge2_recall": np.mean(rouge2_recall),
        "rouge2_f": np.mean(rouge2_f),
        "rougeL_precision": np.mean(rougeL_precision),
        "rougeL_recall": np.mean(rougeL_recall),
        "rougeL_f": np.mean(rougeL_f),
    }

    print("[INFO] Ellipsis corpus scoring complete. Results saved to:", filename)
    return metrics, filename

def score_rudabank(model_name, temperature, n_samples):
    """Run scoring on the rudabank.csv dataset."""
    # Load data
    data = pd.read_csv("data/rudabank.csv")
    # Get random n_samples
    data = data.sample(n=n_samples, random_state=42)
    
    # Get unique tags for options
    unique_tags = sorted(data["tag"].unique())
    options = ", ".join(unique_tags)

    # Initialize model
    model = ChatOpenAI(
        model=model_name, api_key=API_KEY, base_url=BASE_URL, temperature=temperature
    )
    structured_llm = model.with_structured_output(Answer)

    # Generate model answers
    results = []
    print("[INFO] Starting RudaBank scoring...")
    for _, row in tqdm(data.iterrows(), desc="Scoring RudaBank", total=len(data)):
        try:
            res = structured_llm.invoke(
                rudabank_prompt_template.invoke(
                    {
                        "options": options,
                        "initial_utterance": row["initial_utterance"],
                        "tagged_utterance": row["tagged_utterance"],
                    }
                )
            )
            results.append((row, res, row["tag"]))
        except Exception as e:
            print(f"Error processing row {row['id']}: {e}")

    # Calculate metrics
    y_true = [i[2].strip().lower() for i in results]
    y_pred = [i[1].answer.strip().lower() for i in results]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
    }

    # Save answers to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rudabank_{model_name.replace('/','_')}_temp{temperature}_{timestamp}.csv"
    answers_df = pd.DataFrame(
        {
            "task_id": [i[0]["id"] for i in results],
            "answer": [
                i[1]["answer"] if isinstance(i[1], dict) else i[1].answer
                for i in results
            ],
            "correct_answer": [i[2] for i in results],
        }
    )
    answers_df.to_csv(filename, index=False)

    print("[INFO] RudaBank scoring complete. Results saved to:", filename)
    return metrics, filename

def score_disrpt(model_name, temperature):
    """Run scoring on the disrpt.json dataset."""
    # Load data
    with open("data/disrpt.json", "r") as f:
        data = json.load(f)

    # Initialize model
    model = ChatOpenAI(
        model=model_name, api_key=API_KEY, base_url=BASE_URL, temperature=temperature
    )
    structured_llm = model.with_structured_output(Answer)

    # Generate model answers
    results = []
    print("[INFO] Starting disrpt scoring...")
    for i in tqdm([x for x in data.values() if x.get("sent_2")], desc="Scoring disrpt"):
        try:
            res = structured_llm.invoke(
                disrpt_prompt_template.invoke(
                    {
                        "options": i["choices"],
                        "sent_1": i["sent_1"],
                        "sent_2": i["sent_2"],
                    }
                )
            )
            results.append((i, res, i["label"]))
        except Exception as e:
            print(e)

    # Calculate metrics
    y_true = [i[2].strip().lower() for i in results]
    y_pred = [i[1].answer.strip().lower() for i in results]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
    }

    # Save answers to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"disrpt_{model_name.replace('/','_')}_temp{temperature}_{timestamp}.csv"
    answers_df = pd.DataFrame(
        {
            "task_id": [i[0].get("id", idx) for idx, i in enumerate(results)],
            "answer": [
                i[1]["answer"] if isinstance(i[1], dict) else i[1].answer
                for i in results
            ],
            "correct_answer": [i[2] for i in results],
        }
    )
    answers_df.to_csv(filename, index=False)

    print("[INFO] Disrpt scoring complete. Results saved to:", filename)
    return metrics, filename


def extract_answer(raw_answer: str):
    cleaned = re.sub(r"[\"\'{}\[\]\(\)]", "", str(raw_answer).strip())
    if cleaned.isdigit():
        return int(cleaned)
    explicit_match = re.search(
        r"(?:Ответ|ответ|Answer|answer|ans|ANS)[:\s]*(\d+)", cleaned, re.IGNORECASE
    )
    if explicit_match:
        return int(explicit_match.group(1))
    digit_match = re.search(r"\b(\d+)\b", cleaned)
    if digit_match:
        return int(digit_match.group(1))
    binary_match = re.search(r"[^0-9]([01])(?![0-9])", cleaned)
    if binary_match:
        return int(binary_match.group(1))
    return None


def generate_prompt(mode, **kwargs):
    base_instruction = (
        "Инструкция: Внимательно прочитай задание и дай точный ответ в требуемом формате.\n"
        "Твой ответ должен содержать ТОЛЬКО номер правильного варианта в строгом формате:\n"
        '"Ответ: [номер]"\n'
        "Никаких дополнительных объяснений, обоснований или текста после ответа быть не должно.\n"
    )
    if mode == "meaning":
        return (
            base_instruction
            + f"\nЗадание: Определи, какое значение соответствует данному выражению в данном контексте.\nВыражение: {kwargs['idiom']}\nКонтекст: {kwargs['example']}\nВарианты ответа: {kwargs['possible_meanings']}\nОтвет:\n"
        )
    elif mode == "text":
        return (
            base_instruction
            + f"\nЗадание: Определи, в каком тексте выражение означает указанное значение.\nВыражение: {kwargs['idiom']}\nЗначение: {kwargs['current_meaning']}\nТексты: {kwargs['texts']}\nОтвет:\n"
        )
    elif mode == "literal":
        return (
            base_instruction
            + f"\nЗадание: Определи, используется ли выражение в прямом или переносном смысле.\nВыражение: {kwargs['idiom']}\nКонтекст: {kwargs['text']}\nВарианты: 0 - буквальное значение, 1 - переносное значение\nОтвет:\n"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def score_idioms(model, api_key, base_url, n_samples=50):
    """
    Оценивает три задачи по идиомам: literal, meaning, text.
    Возвращает словарь с метриками для каждой задачи.
    """
    # Load all three datasets
    with open("data/idiom_literal.json") as f:
        literal_data = json.load(f)
    with open("data/idiom_two_meanings_2.json") as f:
        meaning_data = json.load(f)
    with open("data/idiom_three_texts.json") as f:
        text_data = json.load(f)

    def ask_model(dct, mode, n_samples):
        sampled_keys = random.sample(list(dct), min(n_samples, len(dct)))
        sampled_data = {k: dct[k] for k in sampled_keys}
        df = pd.DataFrame(columns=["task_id", "idiom", "meaning", "label", "answer"])
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, api_key=API_KEY, base_url=BASE_URL)
        for i, info in tqdm(sampled_data.items(), desc=f"Scoring idioms ({mode})"):
            idiom = info.get("idiom", "")
            current_meaning = info.get("current_meaning", "")
            label = info.get("correct_label", "")
            task_id = i
            try:
                if mode == "meaning":
                    options = info["possible_meanings"]
                    text = info["example"]
                    prompt = generate_prompt(
                        mode, idiom=idiom, example=text, possible_meanings=options
                    )
                elif mode == "text":
                    texts = info["texts"]
                    prompt = generate_prompt(
                        mode, idiom=idiom, current_meaning=current_meaning, texts=texts
                    )
                elif mode == "literal":
                    text = info["text"]
                    prompt = generate_prompt(mode, idiom=idiom, text=text)
                else:
                    continue
                ans = llm.invoke(prompt).content
                df.loc[len(df)] = [task_id, idiom, current_meaning, label, ans]
            except Exception as e:
                print(f"Ошибка: {str(e)}")
        return df

    def count_metrics(df):
        df = df.copy()
        df["ans"] = df["answer"].apply(extract_answer)
        df = df.dropna()
        y_true = df["label"].astype(int)
        y_pred = df["ans"].astype(int)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "report": classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            ),
        }

    # Run all three tasks
    literal_df = ask_model(literal_data, "literal", n_samples)
    meaning_df = ask_model(meaning_data, "meaning", n_samples)
    text_df = ask_model(text_data, "text", n_samples)

    metrics = {
        "literal": count_metrics(literal_df),
        "meaning": count_metrics(meaning_df),
        "text": count_metrics(text_df),
    }

    # Optionally, save answers to CSV
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    literal_csv = f"idioms_literal_{model.replace('/', '_')}_{timestamp}.csv"
    meaning_csv = f"idioms_meaning_{model.replace('/', '_')}_{timestamp}.csv"
    text_csv = f"idioms_text_{model.replace('/', '_')}_{timestamp}.csv"
    literal_df_out = pd.DataFrame(
        {
            "task_id": literal_df["task_id"],
            "answer": literal_df["answer"],
            "correct_answer": literal_df["label"],
        }
    )
    meaning_df_out = pd.DataFrame(
        {
            "task_id": meaning_df["task_id"],
            "answer": meaning_df["answer"],
            "correct_answer": meaning_df["label"],
        }
    )
    text_df_out = pd.DataFrame(
        {
            "task_id": text_df["task_id"],
            "answer": text_df["answer"],
            "correct_answer": text_df["label"],
        }
    )
    literal_df_out.to_csv(literal_csv, index=False)
    meaning_df_out.to_csv(meaning_csv, index=False)
    text_df_out.to_csv(text_csv, index=False)

    with open(f"idioms_{model.replace('/', '_')}_metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(
        "[INFO] Idioms scoring complete. Results saved to:",
        literal_csv,
        meaning_csv,
        text_csv,
    )
    return metrics, [literal_csv, meaning_csv, text_csv]

def score_coref_anaphoric(model_name, temperature):
    # Load data
    with open("data/coref__anaph_ref_choice_questions.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize model
    model = ChatOpenAI(
        model=model_name, api_key=API_KEY, base_url=BASE_URL, temperature=temperature
    )

    # Generate model answers
    print("[INFO] Starting coref 1 scoring...")
    df = pd.DataFrame(columns=['text', 'quest pronoun', 'variants', 'correct_answer', "answer"])
    for q_obj in tqdm(data, desc="Scoring coreference, task 1"):
        ans = model.invoke('Ответь на вопрос по этому фрагменту текста: "' + q_obj['paragraph']["text"] 
            + '"\nТебе нужно понять, к какой сущности относится это упоминание: "' + q_obj['anaphoric span']
            + '". Из предложенных ниже выбери упоминание, которое тоже относится к этой сущности\nВарианты ответа:\n'
            + f'1. {q_obj['variants'][0]};\n2. {q_obj['variants'][1]};\n3. {q_obj['variants'][2]}'
            + '\nНапиши только варинат ответа, 1, 2 или 3, без комментариев и знаков препинания.').content
        df.loc[len(df)] = [q_obj['paragraph']["text"], q_obj['anaphoric span'],  q_obj['variants'], q_obj['gold answer'], ans]

    # Calculate metrics
    y_true = list(df['answer'])
    y_pred = list(df['correct_answer'])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
    }
    # Save answers to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"corefAnaphs_{model_name}_temp{temperature}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    print("[INFO] coref 1 scoring complete. Results saved to:", filename)
    return metrics, filename

def score_is_coref_NPs(model_name, temperature):
    # Load data
    with open("data/coref__are_NPs_coref.json", "r") as f:
        data = json.load(f)
    
    # Initialize model
    model = ChatOpenAI(
        model=model_name, api_key=API_KEY, base_url=BASE_URL, temperature=temperature
    )

    # Generate model answers
    print("[INFO] Starting coref 2 scoring...")
    df = pd.DataFrame(columns=['text', 'first span', 'second span', 'correct_answer', "answer"])
    for q_obj in tqdm(data, desc="Scoring coreference, task 2"):
        ans = model.invoke(f'В тексте: "{q_obj['paragraph']['text']}" упоминания (подстроки) "{q_obj['first']}" и "{q_obj['second']}" отсылают к одной и той же сущности?\n\nОтвечай True, если да, False если нет, без дополнительных комментариев и знаков препинания.')
        if ans == 'true' or ans == 'True': ans = True
        else: ans = False
        df.loc[len(df)] = [q_obj['paragraph']["text"], q_obj['first'],  q_obj['second'], q_obj['gold'], ans]

    # Calculate metrics
    y_true = list([int(bool_value) for bool_value in df['answer']])
    y_pred = list([int(bool_value) for bool_value in df['correct_answer']])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
    }
    # # Save answers to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"corefREs_{model_name}_temp{temperature}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    print("[INFO] coref 2 scoring complete. Results saved to:", filename)
    return metrics, filename

def run_all(model_name: str, temperature: float):
    print("[INFO] Running all scoring tasks...")
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/{model_name.replace('/', '_')}_temp{temperature}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Run all scoring tasks
    rudabank_metrics, rudabank_csv = score_rudabank(model_name, temperature, n_samples=500)
    idioms_metrics, idioms_csvs = score_idioms(model_name, API_KEY, BASE_URL, n_samples=100)
    coref_anaph_metrics, coref_anaph_csv = score_coref_anaphoric(model_name, temperature)
    coref_nps_metrics, coref_nps_csv = score_is_coref_NPs(model_name, temperature)
    ellipsis_metrics, ellipsis_csv = score_ellipsis(model_name, temperature)
    disrpt_metrics, disrpt_csv = score_disrpt(model_name, temperature)
    
    print("[INFO] All tasks complete. Saving metrics...")
    
    # Move all CSV files to the run directory
    all_csvs = [ellipsis_csv, rudabank_csv, disrpt_csv, coref_anaph_csv, coref_nps_csv] + idioms_csvs
    for csv_file in all_csvs:
        if os.path.exists(csv_file):
            os.rename(csv_file, os.path.join(run_dir, os.path.basename(csv_file)))
    
    # Combine metrics
    all_metrics = {
        "ellipsis_corpus": ellipsis_metrics,
        "disrpt_json": disrpt_metrics,
        "idioms": idioms_metrics,
        "rudabank": rudabank_metrics,
        "coref_anaphoric": coref_anaph_metrics,
        "coref_nps": coref_nps_metrics
    }

    # Save metrics to JSON in the run directory
    metrics_filename = os.path.join(run_dir, f"metrics.json")
    with open(metrics_filename, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"[INFO] All metrics and CSVs saved to directory: {run_dir}")
    return all_metrics, run_dir

def calculate_metrics_from_csv(run_dir: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate metrics from CSV files in a run directory.
    
    Args:
        run_dir (str): Path to the run directory containing CSV files
        output_path (Optional[str]): Path to save the table (CSV format). If None, won't save
        
    Returns:
        pd.DataFrame: DataFrame containing the metrics table
    """
    # Find all CSV files in the run directory
    csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {run_dir}")
    
    # Initialize list to store rows
    rows = []
    
    # Process each CSV file
    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Ensure required columns exist
        required_columns = ['answer', 'correct_answer']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Skipping {csv_file} - missing required columns")
            continue
        
        # Clean and prepare the data
        y_true = df['correct_answer'].astype(str).str.strip().str.lower()
        y_pred = df['answer'].astype(str).str.strip().str.lower()
        
        # Get task name from filename
        task_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Calculate metrics
        metrics = {
            'Task': task_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average="macro", zero_division=0),
            'Recall': recall_score(y_true, y_pred, average="macro", zero_division=0),
            'F1': f1_score(y_true, y_pred, average="macro", zero_division=0)
        }
        rows.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Round numeric columns to 3 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    
    # Save to CSV if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[INFO] Metrics table saved to: {output_path}")
    
    return df

def create_metrics_table(metrics_json_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convert metrics JSON into a readable table format.
    
    Args:
        metrics_json_path (str): Path to the metrics JSON file
        output_path (Optional[str]): Path to save the table (CSV format). If None, won't save
        
    Returns:
        pd.DataFrame: DataFrame containing the metrics table
    """
    # Read the JSON file
    with open(metrics_json_path, 'r') as f:
        metrics = json.load(f)
    
    # Initialize list to store rows
    rows = []
    
    # Process each task
    for task_name, task_metrics in metrics.items():
        if task_name == 'idioms':
            # Handle idioms which has nested metrics
            for sub_task, sub_metrics in task_metrics.items():
                row = {
                    'Task': f'idioms_{sub_task}',
                    'Accuracy': sub_metrics['accuracy'],
                    'Precision': sub_metrics['precision'],
                    'Recall': sub_metrics['recall'],
                    'F1': sub_metrics['f1']
                }
                rows.append(row)
        else:
            # Handle other tasks
            row = {
                'Task': task_name,
                'Accuracy': task_metrics['accuracy'],
                'Precision': task_metrics['precision'],
                'Recall': task_metrics['recall'],
                'F1': task_metrics['f1']
            }
            # Add ROUGE metrics if they exist
            if 'rouge1_f' in task_metrics:
                row.update({
                    'ROUGE-1 F1': task_metrics['rouge1_f'],
                    'ROUGE-2 F1': task_metrics['rouge2_f'],
                    'ROUGE-L F1': task_metrics['rougeL_f']
                })
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Round numeric columns to 3 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    
    # Save to CSV if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[INFO] Metrics table saved to: {output_path}")
    
    return df


# Example usage
if __name__ == "__main__":
    model_name = "qwen/qwen3-30b"
    temperature = 0
    
    # Run all tasks and get the run directory
    metrics, run_dir = run_all(model_name, temperature)
    
    # Create metrics table from the run directory
    metrics_table = create_metrics_table(
        os.path.join(run_dir, "metrics.json"),
        os.path.join(run_dir, "metrics_table.csv")
    )
    print("\nMetrics Table:")
    print(metrics_table.to_string(index=False))