import json
from dotenv import load_dotenv

from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

disrpt_prompt_template = PromptTemplate.from_template("Define a connection between two sentences. Possible answers are those:\n {options}.\nSentence 1: {sent_1}\nSentence 2: {sent_2}\nGive only one answer from the options and reason why did you choose that. Use JSON for output consist of two fields: `answer` and `reason`.")

tape_prompt_template = PromptTemplate.from_template("Define a connection between two sentences. Possible answers are those:\n {options}.\nSentence 1: {sent_1}\nSentence 2: {sent_2}\nGive only one answer from the options and reason why did you choose that.")


class Answer(BaseModel):
    """Question answer"""
    answer: str = Field(description="only one word from the possible answers containing your answer.")
    reason: str = Field(description="reason why did you answer this way.")

def score(modelname, data):
    llm = ChatOpenAI(model=modelname)
    structured_llm = llm.with_structured_output(Answer)
    results = []
    for i in tqdm([x for x in data if x['sent_2']][:100]):
        if i['sent_2'] != '':
            try:
                res = structured_llm.invoke(disrpt_prompt_template.invoke({'options': i['choices'],
                                    'sent_1': i['sent_1'],
                                    'sent_2': i['sent_2']}))
                results.append((
                    i, res, i['label']
                ))
            except Exception as e:
                print(e)
            
            
    correct = 0
    for i in results:
        if i[1].answer.lower() == i[2].lower():
            correct += 1

    print(correct/len(results))
            
    return results