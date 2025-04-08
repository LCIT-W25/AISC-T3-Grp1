# import os
# import re
# import base64
# import emoji
# import pickle
# import numpy as np
# import torch
# import nltk
# from io import BytesIO
# from PIL import Image
# from markupsafe import Markup
# from flask import Flask, render_template, request, send_file, jsonify
# from transformers import CLIPTextModel, CLIPTokenizer
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from lime.lime_text import LimeTextExplainer
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# # ========== NLTK Setup ==========
# nltk.download('punkt')
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# # ========== Flask App ==========
# app = Flask(__name__)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== CGAN Setup ==========
# generator = load_model('./Models/c_generator.h5')
# label_mapping = pickle.load(open('./Models/c_label_mapping.pkl', 'rb'))
# reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# # ========== Diffusion Model Setup ==========
# class TextEncoder(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#         self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#     def forward(self, text):
#         tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
#         embeddings = self.clip_model(**tokens).last_hidden_state.mean(dim=1)
#         return embeddings

# class UNet(torch.nn.Module):
#     def __init__(self, text_emb_dim=512):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
#         self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
#         self.conv3 = torch.nn.Conv2d(128, 128, 3, padding=1)
#         self.fc_text = torch.nn.Linear(text_emb_dim, 128)
#         self.final_conv = torch.nn.Conv2d(128, 3, 3, padding=1)

#     def forward(self, x, text_emb):
#         x = torch.nn.functional.relu(self.conv1(x))
#         x = torch.nn.functional.relu(self.conv2(x))
#         x = torch.nn.functional.relu(self.conv3(x))
#         text_emb = self.fc_text(text_emb).view(-1, 128, 1, 1)
#         x = x + text_emb
#         return self.final_conv(x)

# text_encoder = TextEncoder().to(device)
# text_encoder.load_state_dict(torch.load('./Models/text_encoder.pth'))
# text_encoder.eval()

# diffusion_model = UNet(text_emb_dim=512).to(device)
# diffusion_model.load_state_dict(torch.load('./Models/diffusion_model.pth'))
# diffusion_model.eval()

# # ========== Sentiment Classifier Setup ==========
# tokenizer = pickle.load(open('Models/transfor_tokenizer.pkl', 'rb'))
# reverse_sentiment_map = pickle.load(open('Models/transfor_reverse_label_map.pkl', 'rb'))
# sentiment_model = load_model('Models/causal_transformer_model.keras')
# CLASS_NAMES = ['Positive', 'Negative', 'Neutral']
# MAX_LEN = 78

# def handle_emojis(text):
#     return emoji.demojize(text)

# def preprocess_text(text):
#     text = handle_emojis(text)
#     tokens = word_tokenize(text.lower())
#     return ' '.join([stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words])

# def predict_proba(texts):
#     processed = [preprocess_text(t) for t in texts]
#     sequences = tokenizer.texts_to_sequences(processed)
#     padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
#     return sentiment_model.predict(padded)

# # ========== RAG Chatbot Model Setup ==========
# with open("Models/word_to_index.pkl", "rb") as f:
#     word_to_index = pickle.load(f)

# class CausalTransformer(torch.nn.Module):
#     def __init__(self, vocab_size, embedding_dim, max_len, num_heads=8, num_layers=6):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
#         self.position_encoding = torch.nn.Embedding(max_len, embedding_dim)
#         self.transformer = torch.nn.Transformer(d_model=embedding_dim, nhead=num_heads,
#                                                 num_encoder_layers=num_layers, num_decoder_layers=num_layers)
#         self.fc_out = torch.nn.Linear(embedding_dim, vocab_size)

#     def forward(self, query, passage, answer):
#         device = query.device
#         seq_len = query.size(1)
#         pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
#         query_emb = self.embedding(query) + self.position_encoding(pos_ids)
#         passage_emb = self.embedding(passage) + self.position_encoding(pos_ids)
#         answer_emb = self.embedding(answer) + self.position_encoding(pos_ids)
#         transformer_out = self.transformer(query_emb, passage_emb)
#         logits = self.fc_out(transformer_out)
#         return logits

# chatbot_model = CausalTransformer(vocab_size=len(word_to_index), embedding_dim=100, max_len=50, num_heads=5)
# chatbot_model.load_state_dict(torch.load("Models/rag_chatbot_model.pth", map_location=device))
# chatbot_model.to(device)
# chatbot_model.eval()

# def tokenize_and_index(text, word_to_index, max_len=50):
#     tokens = text.lower().split()
#     token_ids = [word_to_index.get(word, word_to_index['<UNK>']) for word in tokens]
#     return token_ids[:max_len] + [word_to_index['<PAD>']] * (max_len - len(token_ids))

# # ========== Routes ==========
# @app.route('/')
# def index():
#     return render_template('cgan.html', label_mapping=label_mapping)

# @app.route('/text-to-image')
# def text_to_image_page():
#     return render_template('text_to_image.html', label_mapping={})

# @app.route('/text-classifier', methods=['GET', 'POST'])
# def classify_text():
#     explanation = None
#     label = None
#     raw_text = ''
#     probs = {}
#     lime_html = None

#     if request.method == 'POST':
#         raw_text = request.form['text']
#         processed_text = preprocess_text(raw_text)

#         sequence = tokenizer.texts_to_sequences([processed_text])
#         padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
#         pred_probs = sentiment_model.predict(padded)[0]
#         pred_label = int(np.argmax(pred_probs))
#         label = reverse_sentiment_map[pred_label]
#         probs = {CLASS_NAMES[i]: round(float(p), 3) for i, p in enumerate(pred_probs)}

#         explainer = LimeTextExplainer(class_names=CLASS_NAMES)
#         exp = explainer.explain_instance(
#             raw_text,
#             lambda texts: predict_proba(texts),
#             num_features=10,
#             top_labels=3
#         )
#         explanation = exp.as_list(label=pred_label)
#         lime_html = Markup(re.sub(r"<!DOCTYPE html>", "", exp.as_html()))

#     return render_template('index.html', label=label, explanation=explanation,
#                            text=raw_text, probs=probs, lime_html=lime_html)

# @app.route('/generate_image', methods=['POST'])
# def generate_cgan_image():
#     selected_label = int(request.form['label'])
#     noise = np.random.normal(0, 1, (1, 128))
#     labels = np.array([[selected_label]])
#     gen_imgs = generator.predict([noise, labels])
#     gen_imgs = 0.5 * gen_imgs + 0.5
#     img = (gen_imgs[0] * 255).astype(np.uint8)
#     img_pil = Image.fromarray(img)
#     img_io = BytesIO()
#     img_pil.save(img_io, 'PNG')
#     img_io.seek(0)
#     return send_file(img_io, mimetype='image/png')

# @app.route('/generate', methods=['POST'])
# @torch.no_grad()
# def generate_diffusion_image():
#     data = request.get_json()
#     text = data.get('text', '')
#     x = torch.randn((1, 3, 128, 128)).to(device)
#     text_emb = text_encoder([text])
#     for _ in range(20):
#         x = diffusion_model(x, text_emb)
#     x = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#     x = (x - x.min()) / (x.max() - x.min())
#     img = Image.fromarray((x * 255).astype(np.uint8))
#     img_bytes = BytesIO()
#     img.save(img_bytes, format='PNG')
#     img_bytes.seek(0)
#     img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
#     return jsonify({"message": "Image generated successfully", "image": img_base64})

# @app.route('/rag-chatbot')
# def rag_chatbot():
#     return render_template('rag.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     query = data.get('query', '')
#     passage = data.get('passage', '')
#     answer = data.get('answer', '')

#     query_tensor = torch.tensor([tokenize_and_index(query, word_to_index)], dtype=torch.long).to(device)
#     passage_tensor = torch.tensor([tokenize_and_index(passage, word_to_index)], dtype=torch.long).to(device)
#     answer_tensor = torch.tensor([tokenize_and_index(answer, word_to_index)], dtype=torch.long).to(device)

#     with torch.no_grad():
#         output = chatbot_model(query_tensor, passage_tensor, answer_tensor)
#         predicted_ids = output.argmax(dim=-1)[0].tolist()

#     index_to_word = {v: k for k, v in word_to_index.items()}
#     predicted_words = [index_to_word.get(idx, '') for idx in predicted_ids]

#     return jsonify({'predicted_answer': ' '.join(predicted_words)})

# if __name__ == '__main__':
#     app.run(debug=True)


import torch
import torch.nn as nn
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Load Vocabulary =========
with open("Models/word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)
index_to_word = {v: k for k, v in word_to_index.items()}

# ========= Tokenizer =========
def tokenize_and_index(text, word_to_index, max_len=50):
    tokens = text.lower().split()
    token_ids = [word_to_index.get(word, word_to_index['<UNK>']) for word in tokens]
    return token_ids[:max_len] + [word_to_index['<PAD>']] * (max_len - len(token_ids))

# ========= Model =========
class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = nn.Embedding(max_len, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_encoding(pos_ids)
        x = self.transformer(x)
        return self.fc_out(x)

# Load model
chatbot_model = CausalTransformer(
    vocab_size=len(word_to_index),
    embedding_dim=100,
    max_len=50,
    num_heads=5
).to(device)
chatbot_model.load_state_dict(torch.load("Models/rag_chatbot_model.pth", map_location=device))
chatbot_model.eval()

# ========= Chatbot Autoregressive Prediction =========
@app.route('/causal-chatbot')
def causal_chatbot_page():
    return render_template("chatbot.html")  # Create this HTML with input box and submit button

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data.get('text', '')
    input_ids = tokenize_and_index(input_text, word_to_index)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    max_gen_len = 20
    generated = []
    
    for _ in range(max_gen_len):
        seq_len = input_tensor.size(1)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        x = chatbot_model.embedding(input_tensor) + chatbot_model.position_encoding(pos_ids)
        x = chatbot_model.transformer(x)
        logits = chatbot_model.fc_out(x)
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
        generated.append(next_token.item())
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == word_to_index.get('<EOS>', 0):
            break

    predicted_words = [index_to_word.get(idx, '') for idx in generated]
    return jsonify({'response': ' '.join(predicted_words)})

if __name__ == '__main__':
    app.run(debug=True)
