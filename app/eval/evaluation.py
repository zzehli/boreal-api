import asyncio
import csv
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

from app.eval.generate_data import generate, questions, retrieve_context

load_dotenv()

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    model="gpt-4o",
)

# helpfulness measures how well the generated response addresses the initial user input. 
helpfulness_evaluator = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    judge=llm,
)

# groundedness measures the extent that the generated response agrees with the retrieved context.
groundedness_evaluator = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    judge=llm,
)

# retrieval_relevance measures how relevant retrieved context is to an input query.
retrieval_relevance_evaluator = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
    judge=llm,
)

async def process_question(question: str) -> Dict:
    """Process a single question through the evaluation pipeline."""
    # Get context and generate response
    results = await retrieve_context(question)
    context = [result.document.content for result in results]
    response = await generate(results, question)
    
    # Run evaluations
    helpfulness_result = helpfulness_evaluator(
        inputs=question,
        outputs=response,
    )
    
    groundedness_result = groundedness_evaluator(
        context=context,
        outputs=response,
    )
    
    retrieval_result = retrieval_relevance_evaluator(
        inputs=question,
        context=context,
    )
    
    return {
        "question": question,
        "response": response,
        "helpfulness_score": helpfulness_result.get("score", None),
        "helpfulness_reasoning": helpfulness_result.get("comment", ""),
        "groundedness_score": groundedness_result.get("score", None),
        "groundedness_reasoning": groundedness_result.get("comment", ""),
        "retrieval_score": retrieval_result.get("score", None),
        "retrieval_reasoning": retrieval_result.get("comment", ""),
    }

async def process_questions_batch(questions_batch: List[str]) -> List[Dict]:
    """Process a batch of questions concurrently."""
    tasks = [process_question(q) for q in questions_batch]
    return await asyncio.gather(*tasks)

def calculate_percentages(results: List[Dict]) -> Dict[str, float]:
    """Calculate percentage of True scores for each metric."""
    total = len(results)
    if total == 0:
        return {"helpfulness": 0, "groundedness": 0, "retrieval": 0}
    
    helpfulness_true = sum(1 for r in results if r["helpfulness_score"])
    groundedness_true = sum(1 for r in results if r["groundedness_score"])
    retrieval_true = sum(1 for r in results if r["retrieval_score"])
    
    return {
        "helpfulness": (helpfulness_true / total) * 100,
        "groundedness": (groundedness_true / total) * 100,
        "retrieval": (retrieval_true / total) * 100
    }

async def main():
    # Process questions in batches of 2
    batch_size = 1
    all_results = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(questions) + batch_size - 1)//batch_size}")
        batch_results = await process_questions_batch(batch)
        print("batch_results: ", batch_results)
        all_results.extend(batch_results)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"evaluation_results_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['question', 'response', 'helpfulness_score', 'helpfulness_reasoning',
                     'groundedness_score', 'groundedness_reasoning', 'retrieval_score', 'retrieval_reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Results saved to {csv_filename}")
    
    # Calculate and print percentages
    percentages = calculate_percentages(all_results)
    print("\nScore Percentages:")
    print(f"Helpfulness: {percentages['helpfulness']:.1f}%")
    print(f"Groundedness: {percentages['groundedness']:.1f}%")
    print(f"Retrieval: {percentages['retrieval']:.1f}%")

def test_evaluation():
    question = questions[0]
    print("question: ", question)
    results = asyncio.run(retrieve_context(question))
    context = [result.document.content for result in results]
    print("context: ", context)
    response = asyncio.run(generate(results, question))
    helpfulness_result = helpfulness_evaluator(
        inputs=question,
        outputs=response,
    )
    print("helpfulness_result: ", helpfulness_result)
    groundedness_result = groundedness_evaluator(
        context=context,
        outputs=response,
    )
    print("groundedness_result: ", groundedness_result)
    retrieval_result = retrieval_relevance_evaluator(
        inputs=question,
        context=context,
    )
    print("retrieval_result: ", retrieval_result)

if __name__ == "__main__":
    # test_evaluation()
    asyncio.run(main())