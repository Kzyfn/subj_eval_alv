## coding: UTF-8
# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for
import numpy as np

from speech_model import synthesize, randomname
#synthesize は音声ファイルのパスを返す

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)

# メッセージをランダムに表示するメソッド
def picked_up():
    messages = [
        "こんにちは、あなたの名前を入力してください",
        "やあ！お名前は何ですか？",
        "あなたの名前を教えてね"
    ]
    # NumPy の random.choice で配列からランダムに取り出し
    return np.random.choice(messages)

# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理


@app.route('/')
def index():
    title = "ようこそ"
    message = picked_up()
    target_sentence = 'みずをマレーシアから買わなくてはならないのです'

    sentence_form = []
    for i, letter in enumerate(target_sentence):
        letter_form = "<label>{}</label>".format(letter)
        for j in range(4):
            tmp = '<input type="radio" name={} value={} required>'.format(i, j)
            letter_form += tmp

        sentence_form.append(letter_form)

    # index.html をレンダリングする
    return render_template('index.html',
                           message=message, title=title, target_sentence=target_sentence, sentence_form = sentence_form)

# /post にアクセスしたときの処理
@app.route('/post', methods=['GET', 'POST'])
def post():
    title = "こんにちは"
    target_sentence = 'みずをマレーシアから買わなくてはならないのです'
    if request.method == 'POST':
        # リクエストフォームから「名前」を取得して

        z = []
        for i in range(23):
            z.append(request.form[str(i)])

        sentence_form = []
        for i, letter in enumerate(target_sentence):
            letter_form = '<label style="margin-bottom: 1rem;">{}</label>'.format(letter)
            for j in range(4):
                checked = 'checked' if z[i] == str(j) else ''
                tmp = '<input type="radio" name={} value={} required {} style="margin-bottom: 1rem;">'.format(i, j, checked)
                letter_form += tmp

            sentence_form.append(letter_form)

        speech_filepath = synthesize(z)
        #speech_filepath='../wav/sample.wav'
        target_sentence = 'みずをマレーシアから買わなくてはならないのです'

        return render_template('index.html',
                             title=title, filepath = speech_filepath, target_sentence=target_sentence, sentence_form=sentence_form)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))


@app.route('/done', methods=['GET', 'POST'])
def done():
    if request.method == 'POST':
        # リクエストフォームから「名前」を取得して

        z = []
        for i in range(23):
            z.append(int(request.form[str(i)]))

        #speech_filepath = synthesize(z)
        result_path = randomname(10)
        np.savetxt('static/results/{}.csv'.format(result_path), z)

        return render_template('done.html', result_path = result_path)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0') # どこからでもアクセス可能に
