import streamlit as st
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
    AutoModelForCausalLM,
)
from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image

# Configure environment for better CUDA memory allocation (if available)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.9,max_split_size_mb:512'

# =============================
# Load models
# =============================
@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

@st.cache_data
def load_story_predictor():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")


@st.cache_resource
def load_question_answering_model():
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def load_chatbot_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def load_image_generation_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to("cpu")  # Ensure it runs on CPU
    return pipe


# =============================
# Text Summarization Function
# =============================
def summarize(text, model, tokenizer, max_length=130, min_length=30):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# =============================
# GPT-2 Next word Prediction Function
# =============================
def predict_next_word(prompt, model, tokenizer, top_k=5):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]
    top_k_tokens = torch.topk(next_token_logits, top_k).indices[0].tolist()
    predicted_tokens = [tokenizer.decode([token]) for token in top_k_tokens]
    return predicted_tokens

# =============================
# Story prediction Function
# =============================

@st.cache_data
def generate_story(prompt, max_length=200):
    story_predictor = load_story_predictor()
    return story_predictor(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]


# =============================
# Sentiment Analysis Function
# =============================
def analyze_sentiment(texts, sentiment_pipeline):
    results = sentiment_pipeline(texts)
    return results


# =============================
# Question Answering Function
# =============================
def answer_question(question, context, model, tokenizer):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


# =============================
# Chatbot Function
# =============================
def chat_with_model(prompt, chat_history_ids, model, tokenizer):
    new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = (
        torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    )
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids


# =============================
# Image Generation Function
# =============================
def generate_image(prompt, pipe):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image


# =============================
# Streamlit App UI
# =============================
st.title("Multifunctional NLP & Image Generation App")
st.subheader("Explore text summarization, next word prediction, Story prediction,sentiment analysis, question answering, chatbot interaction, and image generation using Hugging Face model.")


# Path to the downloaded image (update the path as needed)
image_path = r"C:\Users\hbhat\OneDrive\Desktop\Huggingface-ai.png"  # Replace with the actual file path

# Open and display the image
image = Image.open(image_path)
st.image(image, caption=None, use_column_width=True)

# #Side bar styling
# st.markdown(
#     """
#     <style>
#     /* Target the selectbox inside the sidebar */
#     .css-1n76v7u {
#         font-size: 100px;
#         padding: 50px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# Sidebar options
task = st.sidebar.selectbox(
    "Choose a task",
    ["Text Summarization", "Next Word Prediction", "Story Prediction", "Sentiment Analysis", "Question Answering", "Chatbot", "Image Generation"],
)



# Load models
bart_tokenizer, bart_model = load_summarization_model()
gpt2_tokenizer, gpt2_model = load_gpt2_model()
sentiment_pipeline = load_sentiment_pipeline()
qa_tokenizer, qa_model = load_question_answering_model()
chatbot_tokenizer, chatbot_model = load_chatbot_model()
image_pipe = load_image_generation_model()

# Summarization
if task == "Text Summarization":
    st.header("Text Summarization")
    input_text = st.text_area("Enter text to summarize", height=200)
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summary = summarize(input_text, bart_model, bart_tokenizer)
            st.subheader("Summary")
            st.write(summary)

# Next Word Prediction
elif task == "Next Word Prediction":
    st.header("Next Word Prediction")
    prompt = st.text_input("Enter a sentence")
    if st.button("Predict Next Words"):
        with st.spinner("Predicting next words..."):
            predicted_words = predict_next_word(prompt, gpt2_model, gpt2_tokenizer)
            st.subheader("Predicted Words")
            st.write(predicted_words)

elif task == "Story Prediction":
    prompt = st.text_input("Enter a prompt to generate a story:")
    if st.button("Generate Story"):
        if prompt.strip():
            with st.spinner("Generating story..."):
                story = generate_story(prompt)
            st.success("Generated Story:")
            st.write(story)
        else:
            st.warning("Please enter a prompt.")



# Sentiment Analysis
elif task == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    input_text = st.text_area("Enter text(s) for sentiment analysis (separate multiple texts with newlines)", height=200)
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            texts = input_text.split("\n")
            results = analyze_sentiment(texts, sentiment_pipeline)
            st.subheader("Sentiment Analysis Results")
            for text, result in zip(texts, results):
                st.write(f"**Text**: {text}")
                st.write(f"**Sentiment**: {result['label']}, **Score**: {result['score']:.4f}")

# Question Answering
elif task == "Question Answering":
    st.header("Question Answering")
    context = st.text_area("Enter context", height=200)
    question = st.text_input("Enter your question")
    if st.button("Get Answer"):
        with st.spinner("Getting answer..."):
            answer = answer_question(question, context, qa_model, qa_tokenizer)
            st.subheader("Answer")
            st.write(answer)

# Chatbot
elif task == "Chatbot":
    st.header("Chatbot")
    chat_history_ids = None
    user_input = st.text_input("You: ")
    if st.button("Chat"):
        with st.spinner("Chatting..."):
            response, chat_history_ids = chat_with_model(user_input, chat_history_ids, chatbot_model, chatbot_tokenizer)
            st.subheader("Bot:")
            st.write(response)

# Image Generation
elif task == "Image Generation":
    st.header("Image Generation")
    prompt = st.text_input("Enter a description for the image")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = generate_image(prompt, image_pipe)
            st.image(image, caption="Generated Image")


# Footer
st.markdown("---")
rating = st.slider("Select a rating based on your experience", 0, 3, 5)
st.write("You rated:", rating)
st.markdown("Please share your experience with us.Thank you!")