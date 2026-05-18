INSTALL:

pip install -r requirements.txt

RUN:

python backend/train_tokenizer.py
python backend/preprocess.py
python backend/train.py
python backend/app.py