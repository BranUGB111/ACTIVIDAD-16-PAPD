from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.linear_model import LinearRegression


BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "datos.csv"


class PredictorVentas:
    """Entrena y utiliza un modelo de regresión lineal simple."""

    def __init__(self, archivo_csv: Union[str, Path] = DATASET) -> None:
        self.archivo_csv = Path(archivo_csv)
        self._modelo = LinearRegression()
        self._entrenado = False

    def cargar_datos(self) -> pd.DataFrame:
        return pd.read_csv(self.archivo_csv)

    def entrenar(self) -> None:
        datos = self.cargar_datos()
        entradas = datos.loc[:, ["publicidad"]]
        salidas = datos.loc[:, "ventas"]
        self._modelo.fit(entradas, salidas)
        self._entrenado = True

    def estimar_ventas(self, inversion_publicidad: float) -> float:
        if not self._entrenado:
            self.entrenar()

        valor = pd.DataFrame({"publicidad": [inversion_publicidad]})
        prediccion = self._modelo.predict(valor)[0]
        return round(float(prediccion), 2)


def crear_predictor() -> PredictorVentas:
    predictor = PredictorVentas()
    predictor.entrenar()
    return predictor


if __name__ == "__main__":
    predictor = crear_predictor()
    print(f"Predicción para 120 unidades de publicidad: {predictor.estimar_ventas(120)}")
