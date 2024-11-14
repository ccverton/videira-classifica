import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import image
import numpy as np
import pandas as pd
import plotly.express as px 

@st.cahse_resource # Carrega o processo uma única vez. Sem ele o modelo seria carregado várias vezes.
# A funcionalidade @st.cache_resource do Streamlit é projetada para armazenar em cache recursos que demandam tempo ou 
# processamento para serem carregados, como modelos de aprendizado de máquina.


def carrega_modelo():
   #https://drive.google.com/file/d/1vF0cr_GHL_9Imye7HEtXtT9NQ8S-CanZ/view?usp=drive_link
    url = "https://drive.google.com/file/d/1vF0cr_GHL_9Imye7HEtXtT9NQ8S-CanZ"

    gdown.download(url, "modelo_tflite_quantizado16bits.tflite")
    interpreter = tf.lite.Interpreter(model_path = "modelo_tflite_quantizado16bits.tflite")
    interpreter.allocate_ternsors()

    return interpreter

def carrega_image():
    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png','jpg','jpeg'])

    # Se a variável não estiver vazia faremos algo
    if upload_file is not None:
        image_data = upload_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success("Imagem carregada com sucesso!")

        image = np.array(image, dtype = np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis = 0)

    return image

def previsao (interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])

    classes = ['HealthyGrapes', 'LeafBlight', 'BlackRot', 'BlackMeasles']

    df = pd.DataFrame()
    df["classes"] = classes
    df["probabilidades (%)"] = 100 * output_data[0]

    fig = px.bar(df, y = "classes", x = "probabilidades (%)",orientation='h', text='probabilidades (%)', title='Probabilidade de Classes de Doenças em Uvas')

    st.plotly_chart(fig)

def main():

    st.set_page_config(
    page_title = "Classifica folhas de videira",
    page_icon = ""   
    )

    st.write("Classifica folhas de videira! ")

    # Carrega modelo
    interpreter = carrega_modelo()

    # Carrega imagem
    image = carrega_imagem()

    # Classifica imagem
    if image is not None:

        previsao(interpreter, image)

if __name__=="__main__":
    main()