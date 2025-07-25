from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import shutil
import os
from datetime import datetime, timedelta
from pydantic import BaseModel
from PIL import Image
from controlnet_aux import OpenposeDetector

# Para stable diffusion:
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

# Para concatenar
import cv2

#CONFIGURACIONES
SECRET_KEY = "secretotesis"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()

#Permitir acceso a la cuenta desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Creacion de la base de datos
Base = declarative_base()
engine = create_engine("sqlite:///C:/Users/alexi/Desktop/tesis_2025/mi_api/tesis.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

#Generar token para autorizar las funciones
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

#MODELOS DE LA BASE DE DATOS
class Usuario(Base):
    __tablename__ = "usuarios"
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, unique=True, index=True)
    contrasena = Column(String)

class Promt(Base):
    __tablename__ = "promts_personalizados"
    id = Column(Integer, primary_key=True, index=True)
    categoria = Column(String)
    nombre = Column(String)
    promt = Column(String)
    usuario_id = Column(Integer)

Base.metadata.create_all(bind=engine)

#FUNCION PARA DARLE UN TOKEN AL USUARIO CREADO
def crear_token(datos: dict):
    datos_copia = datos.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    datos_copia.update({"exp": expire})
    return jwt.encode(datos_copia, SECRET_KEY, algorithm=ALGORITHM)


#FUNCION PARA REVISAR LOS TOKEN EN CADA FUNCION
def obtener_usuario_actual(token: str = Depends(oauth2_scheme)):
    credenciales = HTTPException(status_code=401, detail="Token inválido")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        nombre_usuario = payload.get("sub")
        if nombre_usuario is None:
            raise credenciales
    except JWTError:
        raise credenciales
    usuario = db.query(Usuario).filter(Usuario.nombre == nombre_usuario).first()
    if usuario is None:
        raise credenciales
    return usuario

#ENDPOINTS DE USUARIO
@app.post("/registro")
def registro(nombre: str = Form(...), contrasena: str = Form(...)):
    if db.query(Usuario).filter_by(nombre=nombre).first():
        raise HTTPException(status_code=400, detail="Usuario ya existe")
    hash = pwd_context.hash(contrasena)
    nuevo = Usuario(nombre=nombre, contrasena=hash)
    db.add(nuevo)
    db.commit()
    return {"mensaje": "Usuario registrado"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    usuario = db.query(Usuario).filter(Usuario.nombre == form_data.username).first()
    if not usuario or not pwd_context.verify(form_data.password, usuario.contrasena):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    token = crear_token({"sub": usuario.nombre})
    return {"access_token": token, "token_type": "bearer"}

#FOTOGRAMAS
@app.post("/subir_fotograma")
def subir_fotograma(file: UploadFile = File(...), usuario: Usuario = Depends(obtener_usuario_actual)):
    carpeta = f"imagenes/fotogramas/{usuario.nombre}"
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, file.filename)
    with open(ruta, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"mensaje": f"Fotograma subido para usuario {usuario.nombre}"}

@app.get("/fotogramas")
def listar_fotogramas(usuario: Usuario = Depends(obtener_usuario_actual)):
    carpeta = f"imagenes/fotogramas/{usuario.nombre}"
    if not os.path.exists(carpeta):
        return []
    return [f"http://127.0.0.1:8000/imagenes/fotogramas/{usuario.nombre}/{archivo}" for archivo in os.listdir(carpeta)]

#GENERAR ESQUELETOS SOLO DE LOS SELECCIONADOS
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

class ArchivosSeleccionados(BaseModel):
    nombres: List[str]

@app.post("/generar_esqueletos")
def generar_esqueletos_archivos(data: ArchivosSeleccionados, usuario: Usuario = Depends(obtener_usuario_actual)):
    carpeta_entrada = f"imagenes/fotogramas/{usuario.nombre}"
    carpeta_salida = f"imagenes/esqueletos/{usuario.nombre}"

    if not os.path.exists(carpeta_entrada):
        raise HTTPException(status_code=404, detail="No se encontraron fotogramas para este usuario.")

    os.makedirs(carpeta_salida, exist_ok=True)

    generados = 0

    for nombre in data.nombres:
        ruta_original = os.path.join(carpeta_entrada, nombre)
        if not os.path.exists(ruta_original):
            continue

        imagen = Image.open(ruta_original).convert("RGB")
        imagen_pose = openpose(imagen)
        ruta_salida = os.path.join(carpeta_salida, f"skeleton_{nombre}")
        imagen_pose.save(ruta_salida)
        generados += 1

    return {"mensaje": "Esqueletos generados", "cantidad": generados}

#PROMTS
@app.post("/promts")
def crear_promt(categoria: str = Form(...), nombre: str = Form(...), promt: str = Form(...), usuario: Usuario = Depends(obtener_usuario_actual)):
    nuevo = Promt(categoria=categoria, nombre=nombre, promt=promt, usuario_id=usuario.id)
    db.add(nuevo)
    db.commit()
    return {"mensaje": "Promt creado"}

@app.get("/promts")
def ver_promts(usuario: Usuario = Depends(obtener_usuario_actual)):
    promts = db.query(Promt).filter(Promt.usuario_id == usuario.id).all()
    return [{"id": p.id, "categoria": p.categoria, "nombre": p.nombre, "promt": p.promt} for p in promts]

@app.delete("/promts/{promt_id}")
def eliminar_promt(promt_id: int, usuario: Usuario = Depends(obtener_usuario_actual)):
    promt = db.query(Promt).filter_by(id=promt_id, usuario_id=usuario.id).first()
    if not promt:
        raise HTTPException(status_code=404, detail="Promt no encontrado")
    db.delete(promt)
    db.commit()
    return {"mensaje": "Promt eliminado"}

#VER ILUSTRACIONES
@app.get("/ilustraciones")
def listar_ilustraciones(usuario: Usuario = Depends(obtener_usuario_actual)):
    carpeta = f"imagenes/ilustraciones/{usuario.nombre}"
    if not os.path.exists(carpeta):
        return []
    return [f"http://127.0.0.1:8000/imagenes/ilustraciones/{usuario.nombre}/{archivo}" for archivo in os.listdir(carpeta)]

#GENERAR ILUSTRACION CON STABLE DIFFUSION + CONTROLNET
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

class IlustracionRequest(BaseModel):
    nombres: List[str]        # nombres de esqueletos seleccionados
    prompt_nombre: str        # nombre del prompt guardado

@app.post("/generar_ilustracion")
def generar_ilustracion(data: IlustracionRequest, usuario: Usuario = Depends(obtener_usuario_actual)):
    carpeta_esqueletos = f"imagenes/esqueletos/{usuario.nombre}"
    carpeta_salida = f"imagenes/ilustraciones/{usuario.nombre}"
    os.makedirs(carpeta_salida, exist_ok=True)

    prompt_db = db.query(Promt).filter(Promt.usuario_id == usuario.id, Promt.nombre == data.prompt_nombre).first()
    if not prompt_db:
        raise HTTPException(status_code=404, detail="Prompt no encontrado")

    prompt_dinamico = prompt_db.promt
    resultados = []

    for nombre_img in data.nombres:
        ruta_esqueleto = os.path.join(carpeta_esqueletos, nombre_img)
        if not os.path.exists(ruta_esqueleto):
            continue

        imagen_pose = Image.open(ruta_esqueleto).convert("RGB")
        result = pipe(prompt_dinamico, image=imagen_pose, num_inference_steps=13).images[0]

        base_nombre = f"sd_{os.path.splitext(nombre_img)[0]}"
        extension = ".png"
        nombre_salida = base_nombre + extension
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        contador = 1
        while os.path.exists(ruta_salida):
            nombre_salida = f"{base_nombre}_{contador}{extension}"
            ruta_salida = os.path.join(carpeta_salida, nombre_salida)
            contador += 1

        result.save(ruta_salida)

        resultados.append(f"http://127.0.0.1:8000/imagenes/ilustraciones/{usuario.nombre}/{nombre_salida}")

    if not resultados:
        raise HTTPException(status_code=400, detail="No se generó ninguna ilustración")

    return {"mensaje": "Ilustraciones generadas correctamente", "urls": resultados}


#MOSTRAR ARCHIVOS (para que se vean en el navegador)
@app.get("/imagenes/{tipo}/{usuario}/{archivo}")
def mostrar_archivo(tipo: str, usuario: str, archivo: str):
    ruta = f"imagenes/{tipo}/{usuario}/{archivo}"
    if not os.path.exists(ruta):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(ruta)

#CONCATENAR IMAGENES

@app.post("/concatenar_esqueletos")
def concatenar_esqueletos(data: ArchivosSeleccionados, usuario: Usuario = Depends(obtener_usuario_actual)):
    carpeta = f"imagenes/esqueletos/{usuario.nombre}"
    rutas = [os.path.join(carpeta, nombre) for nombre in data.nombres]

    # Leer imágenes válidas
    imagenes = [cv2.imread(ruta) for ruta in rutas if os.path.exists(ruta)]
    imagenes = [img for img in imagenes if img is not None]

    if not imagenes:
        raise HTTPException(status_code=400, detail="No se pudieron leer las imágenes")

    # Usar la altura mínima para redimensionar todas
    altura_min = min(img.shape[0] for img in imagenes)
    imagenes_redimensionadas = [
        cv2.resize(img, (int(img.shape[1] * altura_min / img.shape[0]), altura_min))
        for img in imagenes
    ]

    resultado = cv2.hconcat(imagenes_redimensionadas)

    # Crear nombre de archivo de salida único
    base_nombre = "concatenado"
    extension = ".png"
    nombre_final = f"{base_nombre}{extension}"
    ruta_final = os.path.join(carpeta, nombre_final)
    contador = 1
    while os.path.exists(ruta_final):
        nombre_final = f"{base_nombre}_{contador}{extension}"
        ruta_final = os.path.join(carpeta, nombre_final)
        contador += 1

    cv2.imwrite(ruta_final, resultado)

    return {
        "mensaje": "Imagen concatenada creada",
        "url": f"http://127.0.0.1:8000/imagenes/esqueletos/{usuario.nombre}/{nombre_final}"
    }
