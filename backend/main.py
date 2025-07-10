from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # Para permitir la comunicación con el frontend
import cv2
import numpy as np
import io

app = FastAPI()

# --- Configuración de CORS ---
# Esto es CRUCIAL para que tu frontend (que corre en un dominio diferente)
# pueda hablar con tu backend.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Si vas a desplegarlo, añade aquí la URL de tu frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- La Lógica Principal ---
@app.post("/generate-seamless/")
async def create_seamless_texture(file: UploadFile = File(...)):
    """
    Recibe una imagen, la convierte en seamless usando una mezcla de técnicas
    y la devuelve.
    """
    # 1. Leer la imagen que subió el usuario
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Lógica de "Seamless" (Versión simplificada y efectiva)
    # En lugar de implementar Wang Tiles desde cero (que es muy complejo),
    # usaremos una técnica muy efectiva que combina desplazamiento y fusión avanzada.
    height, width, _ = img.shape
    
    # Desplaza la imagen para traer las costuras al centro
    shifted_img = np.roll(img, (height // 2, width // 2), axis=(0, 1))

    # Creamos una máscara para el centro (donde están las costuras)
    # y usamos la magia de `seamlessClone` de OpenCV, que implementa Poisson Blending.
    
    # Tomamos el centro de la imagen original (antes de desplazarla)
    center_patch = img[height//4:3*height//4, width//4:3*width//4]
    
    # Creamos una máscara para la región donde vamos a pegar el parche
    mask = 255 * np.ones(center_patch.shape, center_patch.dtype)
    center_point = (width // 2, height // 2)

    # Usamos Poisson Blending para fusionar el centro original sobre la imagen desplazada,
    # eliminando las costuras de forma inteligente.
    # Esta es la función CLAVE.
    seamless_result = cv2.seamlessClone(center_patch, shifted_img, mask, center_point, cv2.MIXED_CLONE)

    # 3. Preparar la imagen para enviarla de vuelta
    _, encoded_img = cv2.imencode('.PNG', seamless_result)
    
    # 4. Devolver la imagen procesada
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")