from flask import Flask, render_template, request

from modelo import crear_predictor


app = Flask(__name__)
predictor_ventas = crear_predictor()


@app.get("/")
def mostrar_formulario():
    return render_template("index.html")


@app.post("/predecir")
def calcular_prediccion():
    dato_formulario = request.form.get("publicidad", "").strip()

    try:
        inversion = float(dato_formulario)
        resultado = predictor_ventas.estimar_ventas(inversion)
        return render_template("index.html", resultado=resultado, publicidad=inversion)
    except ValueError:
        mensaje = "Ingresa un valor numérico válido para publicidad."
        return render_template("index.html", error=mensaje, publicidad=dato_formulario), 400


if __name__ == "__main__":
    app.run(debug=True)
