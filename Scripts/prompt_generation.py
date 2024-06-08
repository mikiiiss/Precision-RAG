# import os
# import openai
# import numpy as np
# import pandas as pd
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv('OPENAI_API_KEY')

# # Function to load embeddings from CSV
# def load_embeddings_from_csv(file_path):
#     df = pd.read_csv(file_path)
#     if "embedding" not in df.columns or "text" not in df.columns:
#         raise KeyError("The 'embedding' or 'text' column is not present in the CSV file.")
#     df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))
#     return df

# # Function to compute cosine similarity using NumPy
# def compute_cosine_similarity(query_embedding, embeddings):
#     query_embedding = np.array(query_embedding)
#     embeddings = np.array([np.array(embed) for embed in embeddings])
#     similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
#     return similarities

# # Function to perform similarity search
# def similarity_search(query, embeddings_df, top_n=5):
#     response = openai.Embedding.create(
#         input=query,
#         model="text-embedding-ada-002"
#     )
#     query_embedding = response['data'][0]['embedding']
#     similarities = compute_cosine_similarity(query_embedding, embeddings_df["embedding"])
#     embeddings_df["similarity"] = similarities
#     return embeddings_df.sort_values(by="similarity", ascending=False).head(top_n)

# # Function to generate prompts using OpenAI's language model
# def generate_prompts(similar_texts, base_query, model="gpt-3.5-turbo-instruct", max_tokens=100):
#     prompts = []
#     for text in similar_texts:
#         prompt = f"Based on the context of '{text}', {base_query}"
#         response = openai.Completion.create(
#             engine=model,
#             prompt=prompt,
#             max_tokens=max_tokens,
#             n=1,
#             stop=None,
#             temperature=0.7
#         )
#         generated_text = response.choices[0].text.strip()
#         prompts.append(generated_text)
#     return prompts

# # Main function to integrate the similarity search and prompt generation
# def main(query, base_query, embeddings_file='new_embeddings.csv', top_n=5):
#     # Load embeddings
#     embeddings_df = load_embeddings_from_csv(embeddings_file)

#     # Perform similarity search
#     results_df = similarity_search(query, embeddings_df, top_n)

#     # Extract the top N similar texts
#     similar_texts = results_df['text'].tolist()

#     # Generate prompts based on the similar texts
#     generated_prompts = generate_prompts(similar_texts, base_query)

#     # Print the generated prompts
#     for idx, prompt in enumerate(generated_prompts, start=1):
#         print(f"Prompt {idx}: {prompt}")

# if __name__ == "__main__":
#     query = "submission"
#     base_query = "please provide more information about submission."
#     main(query, base_query)


# import os
# import openai
# import numpy as np
# import pandas as pd
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv('OPENAI_API_KEY')

# # Function to load embeddings from CSV
# def load_embeddings_from_csv(file_path):
#     df = pd.read_csv(file_path)
#     if "embedding" not in df.columns or "text" not in df.columns:
#         raise KeyError("The 'embedding' or 'text' column is not present in the CSV file.")
#     df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))
#     return df

# # Function to compute cosine similarity using NumPy
# def compute_cosine_similarity(query_embedding, embeddings):
#     query_embedding = np.array(query_embedding)
#     embeddings = np.array([np.array(embed) for embed in embeddings])
#     similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
#     return similarities

# # Function to perform similarity search
# def similarity_search(query, embeddings_df, top_n=5):
#     response = openai.Embedding.create(
#         input=query,
#         model="text-embedding-ada-002"
#     )
#     query_embedding = response['data'][0]['embedding']
#     similarities = compute_cosine_similarity(query_embedding, embeddings_df["embedding"])
#     embeddings_df["similarity"] = similarities
#     return embeddings_df.sort_values(by="similarity", ascending=False).head(top_n)

# # Function to generate question prompts using OpenAI's language model
# def generate_question_prompts(similar_texts, base_query, model="gpt-3.5-turbo-instruct", max_tokens=100):
#     prompts = []
#     for text in similar_texts:
#         # Customized prompt template reflecting the detailed inquiry as discussed
#         prompt = (
#             f"In the realm of enterprise-grade RAG systems and considering the context of '{text}', "
#             f"propose three insightful questions that delve into the practical applications and understanding of Precision RAG, "
#             f"prompt engineering, and its components—like generation, evaluation, data generation, and user interface. "
#             f"Ensure these questions relate to '{base_query}' and can foster discussions on strategic advantages, "
#             f"challenges, and the benefits for businesses engaging with this technology."
#         )
#         response = openai.Completion.create(
#             engine=model,
#             prompt=prompt,
#             max_tokens=max_tokens,
#             n=1,
#             stop=None,
#             temperature=0.7
#         )
#         generated_text = response.choices[0].text.strip()
#         prompts.append(generated_text)
#     return prompts

# # Main function to integrate the similarity search and prompt generation
# def main(query, base_query, embeddings_file='new_embeddings.csv', top_n=5):
#     # Load embeddings
#     embeddings_df = load_embeddings_from_csv(embeddings_file)

#     # Perform similarity search
#     results_df = similarity_search(query, embeddings_df, top_n)

#     # Extract the top N similar texts
#     similar_texts = results_df['text'].tolist()

#     # Generate question prompts based on the similar texts
#     generated_prompts = generate_question_prompts(similar_texts, base_query)

#     # Print the generated prompts
#     for idx, prompt in enumerate(generated_prompts, start=1):
#         print(f"Prompt {idx}: {prompt}")

# if __name__ == "__main__":
#     query = "References"
#     base_query = "explain about References"
#     main(query, base_query)


# import os
# import openai
# from similarity_search import load_embeddings_from_csv, compute_cosine_similarity, similarity_search
# import numpy as np
# import pandas as pd
# from dotenv import load_dotenv

# # Load environment variables (ensuring this is not redundant if also done in similarity_search.py)
# load_dotenv()
# openai.api_key = os.getenv('OPENAI_API_KEY')

# def generate_question_prompts(similar_texts, base_query, model="gpt-3.5-turbo-instruct", max_tokens=100):
#     prompts = []
#     for text in similar_texts:
#         prompt = (
#             f"In the context of '{text}' and considering '{base_query}', "
#             f"propose three insightful questions that explore the practical implications, challenges, "
#             f"and strategic benefits of implementing advanced RAG systems, focusing on precision, "
#             f"prompt engineering, data handling, and UI aspects. These questions should foster "
#             f"a deep dive into how businesses can leverage this technology effectively."
#         )
#         response = openai.Completion.create(
#             engine=model,
#             prompt=prompt,
#             max_tokens=max_tokens,
#             n=1,
#             stop=None,
#             temperature=0.7
#         )
#         generated_text = response.choices[0].text.strip()
#         prompts.append(generated_text)
#     return prompts

# def main(query, base_query, embeddings_file='new_embeddings.csv', top_n=5):
#     # Load embeddings
#     embeddings_df = load_embeddings_from_csv(embeddings_file)

#     # Perform similarity search
#     results_df = similarity_search(query, embeddings_df)
    
#     # Get top N similar texts
#     top_results = results_df.head(top_n)['text'].tolist()

#     # Generate question prompts based on the similar texts
#     generated_prompts = generate_question_prompts(top_results, base_query)

#     # Print the generated prompts
#     for idx, prompt in enumerate(generated_prompts, start=1):
#         print(f"Prompt {idx}: {prompt}")

# if __name__ == "__main__":
#     query = "References"
#     base_query = "Explain the role of references in knowledge management systems"
#     main(query, base_query)


from similarity_search import load_embeddings_from_csv, compute_cosine_similarity, similarity_search
import os
import openai
from similarity_search import load_embeddings_from_csv, compute_cosine_similarity, similarity_search
import numpy as np
import pandas as pd
from dotenv import load_dotenv
#Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Sample prompts and ground truths for inspiration
EXAMPLE_PROMPTS = [
    {
        "prompt": "What are the core tasks for this week's challenge in the context of {context}?",
        "ground_truth": "The core tasks include understanding specific aspects of {context}, such as {detail1}, {detail2}, and planning for {objective}."
    },
    {
                "prompt": "What is the minimum requirement for the interim submission in the challenge?",
                "ground_truth": "The minimum requirement is to have a well-structured repository with some coding progress made."
            },
            {
                "prompt": "What is the key performance indicator for Tuesday's session on RAG components?",
                "ground_truth": "The key performance indicators include understanding prompt ranking, understanding prompt matching, and ability to reuse previous knowledge."
            },
            {
                "prompt": "What is the instruction for Automatic Prompt Engineering Fundamental Tasks?",
                "ground_truth": "The core tasks include understanding prompt engineering tools and concepts, familiarizing with language models, developing a plan for prompt generation and testing, setting up a development environment, designing a user interface for prompt system, planning integration of LLMs, building and refining prompt generation system, developing automatic evaluation data generation system, implementing prompt testing and evaluation mechanism, and refining and optimizing system based on feedback."
            },
            {
                "prompt": "What is the deadline for the final submission in the challenge?",
                "ground_truth": "The final submission deadline is Saturday 8pm UTC."
            }
    # ... other examples ...
]

def generate_custom_prompts(similar_texts, base_query, model="gpt-3.5-turbo-instruct", max_tokens=100):
    custom_prompts = []
    for example in EXAMPLE_PROMPTS:
        for text in similar_texts:
            # Customize the prompt using the example and context from similarity search results
            prompt = example["prompt"].format(context=text, detail1="precision RAG", detail2="prompt engineering")
            # For simplicity, we're not using the ground_truth here but you could incorporate it in your workflow
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=0.7
            )
            generated_prompt = response.choices[0].text.strip()
            custom_prompts.append({
                "original_prompt": prompt,
                "generated_prompt": generated_prompt
            })
    return custom_prompts

def main(query, base_query, embeddings_file='new_embeddings.csv', top_n=5):
    # Load embeddings and perform similarity search
    embeddings_df = load_embeddings_from_csv(embeddings_file)
    results_df = similarity_search(query, embeddings_df)
    top_results = results_df.head(top_n)['text'].tolist()
    
    # Generate custom prompts based on the similarity search results
    generated_prompts = generate_custom_prompts(top_results, base_query)

    # Print the generated prompts
    for prompt_data in generated_prompts:
        print(f"Original Prompt: {prompt_data['original_prompt']}")
        print(f"Generated Prompt: {prompt_data['generated_prompt']}\n")

if __name__ == "__main__":
    query = "References in Precision RAG systems"
    base_query = "Explain the importance of references in the context of Precision Retrieval-Augmentation Generation systems"
    main(query, base_query)