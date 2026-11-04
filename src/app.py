import base64
from io import BytesIO

from flask import Flask, render_template, request

from classifier import classify_image, load_image


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/classify")
def upload_file():
    upload = request.files.get("image")

    if upload is None or not upload.filename:
        return render_template(
            "index.html",
            error="Choose an image before classifying"
        ), 400

    try:
        image = load_image(upload.stream)
    except ValueError as error:
        return render_template(
            "index.html",
            error=str(error)
        ), 400

    try:
        result = classify_image(image)
    except FileNotFoundError:
        app.logger.exception("Model file is missing")
        return render_template(
            "index.html",
            error=("The model is unavailable. "
                   "Add the trained model to the models folder.")
        ), 503
    except Exception:
        app.logger.exception("Classification failed")
        return render_template(
            "index.html",
            error="Classification failed. Please check the server log."
        ), 500

    image.thumbnail((1000, 1000))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    preview = base64.b64encode(buffer.getvalue()).decode("ascii")

    return render_template(
        "index.html",
        result=result,
        preview=preview
    )


@app.errorhandler(413)
def upload_too_large(error):
    return render_template(
        "index.html",
        error="The upload is too large. Maximum size: 10 MB."
    ), 413


if __name__ == "__main__":
    app.run()
