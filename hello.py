import random, string


def randomname(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)


from flask import Flask, request, render_template, redirect
app = Flask(__name__)
import requests

@app.route('/')
def hello():
    string = randomname(40)
    name = string#"Hello Worl!!d"

    a = '<form action="/regist" method="POST">\n<label><input type="checkbox" name="fav" value="1">ラーメン</label>\n<label><input type="checkbox" name="fav" value="2">うな丼</label>\n<label><input type="checkbox" name="fav" value="3">メロン</label>\n<input type="submit" value="送信"></form>'
    return name + '\n' + a

@app.route('/good')
def good():
    name = "Good"
    return name

@app.route("/regist", methods=['POST'])
def test2():
    favs = request.form.getlist("fav")
    print("favs:", favs) # ['1','2','3']
    return "ok"


if __name__ == "__main__":
    app.run(debug=True)
