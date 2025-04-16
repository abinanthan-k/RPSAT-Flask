from flask import Flask, request, render_template, redirect, url_for, session
from flask_cors import CORS
import os
import time
from services.parser import extract_text_from_pdf, split_into_chunks
from services.chain import split_summaries, prepare_final_summary
from services.closest import return_closest_indices
from services.translator import translate_text

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.secret_key = 'your-secret-key'  # Needed for session

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/result')
def result():
    summary = session.get('summary', 'No summary available.')
    time_taken = session.get('time_taken', 'Unknown')
    return render_template("result.html", summary=summary, time_taken=time_taken)

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files or 'language' not in request.form:
        return "Missing file or language", 400

    file = request.files['file']
    language = request.form['language']
    start = time.time()

    contents = extract_text_from_pdf(file)
    chunks = split_into_chunks(contents)
    selected_indices = return_closest_indices(chunks)
    summaries = split_summaries(selected_indices, chunks)
    final_result = prepare_final_summary(summaries)
    result_text = final_result["output_text"]
    translated = translate_text(result_text, language)
    done_in = round(time.time() - start, 2)

    # Store in session
    session['summary'] = translated
    session['time_taken'] = str(round(done_in, 2))

    return redirect(url_for('result'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
