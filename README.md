Para ejecutar correctamente la API localmente, es necesario tener instalado {Python 3.10.6}, y luego preparar el entorno de trabajo con las siguientes consideraciones, que se detallan a continuación:  
  1. Descarga del Repositorio  
       Descargar el repositorio que se encuentra en GitHub, posteriormente abrir el centro de comandos (CMD) e ingresar a la carpeta del proyecto con el comando: "cd Tesis"  

  2. Instalación de Dependencias  
       Dentro del repositorio se encuentra un archivo \texttt{requirements.txt} con todas las bibliotecas necesarias para ejecutar el sistema. Para instalarlas, se ejecuta: "pip install -r requirements.txt"  
       Al finalizar la instalación ingresa el comando: "pip freeze"  
       Al ejecutar este comando, se deberían visualizar todas las librerías recién instaladas  

  4. Descarga de Modelos Preentrenados  
       Al ejecutar por primera vez el sistema, los modelos preentrenados de ControlNet y Stable Diffusion se descargan automáticamente desde los servidores de HuggingFace:  
         a. "lllyasviel/sd-controlnet-openpose"  
         b. "runwayml/stable-diffusion-v1-5"  
       Es necesario contar con conexión a internet durante esta etapa.  

     
  5. Levantamiento del Servidor Local  
       El sistema utiliza una base de datos SQLite, la cual se configura automáticamente al ejecutar el archivo principal de la API.  
       Para ejecutar la API, se debe ejecutar el siguiente comando en el CMD desde la raíz del proyecto: "uvicorn main:app --reload"  
       Esto levanta el servidor en: "http://127.0.0.1:8000/docs"  

  6. Manual de Usuario  
       La herramienta desarrollada permite a los usuarios cargar imágenes de ejecución de técnicas de artes marciales y generar ilustraciones automáticas a partir de ellos.  
       La herramienta se ejecuta como una API local utilizando FastAPI, y puede ser accedido a través de una interfaz web. Desde esta interfaz, los usuarios autenticados pueden utilizar los distintos endpoints para crear usuarios, iniciar sesión, subir imágenes, gestionar prompts, extraer posiciones corporales en imágenes de esqueletos y generar ilustraciones.  
