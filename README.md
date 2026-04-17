# TDA en series financieras — detección de cambios de régimen

Usar **homología persistente** sobre ventanas deslizantes de precios de múltiples activos para detectar cambios de régimen de mercado (ej. crisis 2008, caída 2020, normalidad).

## Motivación

Los métodos clásicos de detección de cambios de régimen en finanzas usan varianza, correlaciones rolling, HMMs. El Topological Data Analysis (TDA) da un enfoque complementario: mira la **forma** de la nube de puntos en el espacio de retornos y extrae descriptores topológicos (barras de persistencia) que son invariantes ante transformaciones suaves.

Este proyecto aplica homología persistente a ventanas deslizantes de un índice multi-activo y muestra que los descriptores topológicos pican justo antes / durante crisis.

## Stack

- Python 3.11+
- Giotto-TDA (o Ripser)
- Pandas, NumPy
- yfinance (datos de mercado)
- scikit-learn (clasificador sobre features topológicos)
- Matplotlib

## Estructura

```
05-tda-finance/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py           # Descarga y normaliza precios
│   ├── persistence.py    # Homología persistente de ventanas deslizantes
│   ├── features.py       # Persistence landscapes / images
│   └── classify.py       # Clasificador de regímenes
└── notebooks/
```

## Roadmap

- [ ] Descargar precios diarios de índices y sectores (SP500, NASDAQ, sector ETFs).
- [ ] Calcular retornos log y normalizar.
- [ ] Ventanas deslizantes → nubes de puntos en ℝ^d.
- [ ] Homología persistente (dim 0 y 1) por ventana.
- [ ] Convertir diagramas en persistence landscapes / images.
- [ ] Entrenar clasificador (Random Forest) normal vs. crisis.
- [ ] Gráfica temporal con anotaciones de eventos macro reales.

## Resultado esperado

Curva temporal de "topological anomaly score" donde los picos coinciden con: crisis subprime (2008), default europeo (2011), selloff China (2015), COVID (2020). Clasificador con AUC > 0.85 separando régimen normal de crisis.
