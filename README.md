🦙 Fine-Tuning LLM estilo Alpaca con Hugging Face AutoTrain

Este repositorio contiene un script en formato Jupyter Notebook para realizar el fine-tuning de un modelo de lenguaje grande (LLM) utilizando el estilo de entrenamiento de Alpaca mediante la plataforma Hugging Face AutoTrain. Es parte de una propuesta técnica presentada para una posición en el área de inteligencia artificial.

📄 Archivo principal
11_1_autotrain_alpaca.ipynb: Notebook interactivo para configurar y lanzar el fine-tuning en AutoTrain.
11_1_autotrain_alpaca.py: Versión del notebook convertida a script Python ejecutable.

Características principales
Utiliza un dataset estructurado con campos instruction, input y output.

  Preprocesamiento adaptado para formato text2text compatible con AutoTrain.
  Entrenamiento supervisado con configuración mínima y reproducible.
  Integración directa con Hugging Face Hub (sin necesidad de infraestructura local).
  Pensado para experimentación rápida con LLMs de tipo generativo.

Modelo objetivo
Aunque el código permite trabajar con distintos modelos, está optimizado para modelos tipo gpt2, mistral o modelos open-source compatibles con fine-tuning estilo Instruct.

📂 Estructura esperada del dataset
El dataset debe tener el siguiente formato por registro JSON o CSV:

json
Copiar
Editar
{
  "instruction": "Describe the benefits of aquaculture AI.",
  "input": "",
  "output": "AI in aquaculture allows better feed optimization and water quality monitoring."
}
🚀 ¿Cómo usar?
Asegúrate de tener acceso a Hugging Face AutoTrain y haber creado un token de acceso.

Prepara tu dataset en el formato Alpaca.

Ejecuta el notebook paso a paso, autenticando con tu token y cargando el dataset.

Configura los parámetros básicos del proyecto (nombre, tipo, modelo base).

Lanza el entrenamiento desde el entorno AutoTrain.

Requisitos
Cuenta de Hugging Face.
Token de acceso válido con permisos de escritura.
Dataset preformateado (JSON o CSV).
Entorno Python 3.8+ con acceso a Jupyter Notebook.

Aplicación y contexto
Este script se incluye como parte de una postulación para posiciones relacionadas con desarrollo de IA, fine-tuning de LLMs y NLP aplicado a dominios técnicos y científicos, especialmente en contextos donde se requiere eficiencia, escalabilidad y adaptabilidad en la creación de asistentes conversacionales o sistemas instructivos.

 Contacto
Para cualquier consulta técnica o colaboración:

Bruno Alejandro Donayre Donayre
Correo: brunodonayredonadyre@gmail.com


