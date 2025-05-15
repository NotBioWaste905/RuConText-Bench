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

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


class Answer(BaseModel):
    """Question answer"""

    answer: str = Field(
        description="only one word from the possible answers containing your answer."
    )
    reason: str = Field(description="reason why did you answer this way.")


disrpt_prompt_template = PromptTemplate.from_template(
    "Define a connection between two sentences. Possible answers are those:\n {options}.\nSentence 1: {sent_1}\nSentence 2: {sent_2}\nGive only one answer from the options and reason why did you choose that. Use JSON for output consist of two fields: `answer` and `reason`."
)

tape_prompt_template = PromptTemplate.from_template(
    "Define a connection between two sentences. Possible answers are those:\n {options}.\nSentence 1: {sent_1}\nSentence 2: {sent_2}\nGive only one answer from the options and reason why did you choose that."
)


def score_ellipsis(model_name, temperature):
    """Run scoring on the ellipsis corpus dataset."""
    # Load data
    data = pd.read_csv("data/ellipsis.csv")
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
        Ответ дай в формате: изначальное - ответ на 1, эллипсис - ответ на 2, полное - ответ на 3. Ответ должен быть в формате json.""").content

        parsed_ans = json.loads(ans.split("json")[1].strip("```"))
        start.append(parsed_ans["изначальное"])
        with_.append(parsed_ans["эллипсис"])
        final.append(parsed_ans["полное"])

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
    filename = f"ellipsis_{model_name}_temp{temperature}_{timestamp}.csv"
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
    y_true = [i[2].lower() for i in results]
    y_pred = [i[1].answer.lower() for i in results]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
    }

    # Save answers to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"disrpt_{model_name}_temp{temperature}_{timestamp}.csv"
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
    with open("data/idiom_two_meanings.json") as f:
        meaning_data = json.load(f)
    with open("data/idiom_three_texts.json") as f:
        text_data = json.load(f)

    def ask_model(dct, mode, n_samples):
        sampled_keys = random.sample(list(dct), min(n_samples, len(dct)))
        sampled_data = {k: dct[k] for k in sampled_keys}
        df = pd.DataFrame(columns=["task_id", "idiom", "meaning", "label", "answer"])
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
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
    literal_csv = f"idioms_literal_{timestamp}.csv"
    meaning_csv = f"idioms_meaning_{timestamp}.csv"
    text_csv = f"idioms_text_{timestamp}.csv"
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

    with open(f"idioms_metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[INFO] Idioms scoring complete. Results saved to:", literal_csv, meaning_csv, text_csv)
    return metrics, [literal_csv, meaning_csv, text_csv]


def run_all(model_name, temperature):
    print("[INFO] Running all scoring tasks...")
    idioms_metrics, idioms_csvs = score_idioms(
        model_name, API_KEY, BASE_URL, n_samples=50
    )
    disrpt_metrics, disrpt_csv = score_disrpt(model_name, temperature)
    ellipsis_metrics, ellipsis_csv = score_ellipsis(model_name, temperature)
    print("[INFO] All tasks complete. Saving metrics...")
    # Combine metrics
    all_metrics = {
        "ellipsis_corpus": ellipsis_metrics,
        "disrpt_json": disrpt_metrics,
        "idioms": idioms_metrics,
    }

    # Save metrics to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"metrics_{model_name}_temp{temperature}_{timestamp}.json"
    with open(metrics_filename, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("[INFO] All metrics and CSVs saved.")
    # Return all metrics and all CSV filenames
    return all_metrics, [ellipsis_csv, disrpt_csv, *idioms_csvs, metrics_filename]


# Example usage
if __name__ == "__main__":
    model_name = "gpt-4o-mini"
    temperature = 0
    run_all(model_name, temperature)
