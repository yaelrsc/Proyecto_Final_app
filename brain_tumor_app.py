import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms



# Creación de la arquitectura de la red neuronal Convolucional
class ConvNN(torch.nn.Module):

    def __init__(self,activation,nn = 128,dropout_rate = 0.5):
        super().__init__()
        self.activation = activation
        self.normalize1_layer = torch.nn.BatchNorm2d(3)
        # self.normalize2_layer = torch.nn.BatchNorm2d(16)
        self.normalize3_layer = torch.nn.BatchNorm2d(32)
        #self.normalize4_layer = torch.nn.BatchNorm2d(8)

        self.Droput1_layer = torch.nn.Dropout(dropout_rate)
        #self.Droput2_layer = torch.nn.Dropout(0.4)

        
        self.conv1_layer = torch.nn.Conv2d(3,16,kernel_size=3,padding=2)
        self.maxpo1_layer = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2_layer = torch.nn.Conv2d(16,32,kernel_size=3,padding=2)
        self.maxpo2_layer = torch.nn.MaxPool2d(kernel_size=2,stride=2)


        self.flatten_layer = torch.nn.Flatten()

        self.fc1_layer = torch.nn.Linear(103968,nn)
        self.fc2_layer = torch.nn.Linear(nn,4)

    def forward(self,x):

        x = self.normalize1_layer(x)
        x = self.conv1_layer(x)
        x = self.maxpo1_layer(x)
        x = self.activation(x)
        
        
        x = self.conv2_layer(x)
        x = self.maxpo2_layer(x)
        x = self.activation(x)


        x = self.normalize3_layer(x)
        x = self.flatten_layer(x)

        
        x = self.fc1_layer(x)
        x = self.activation(x)
        x = self.Droput1_layer(x)

        x = self.fc2_layer(x)
        x = torch.functional.F.softmax(x)
        
        

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Cargar el modelo guardado

model = ConvNN(torch.nn.ReLU(),nn=164,dropout_rate = 0.782236508846162)
model.load_state_dict(torch.load('modelos/brain_tumor_cnn_model_15.pt')) 
model.to(device)
model.eval()  # Poner el modelo en modo de evaluación

# Definir las transformaciones necesarias (ajustar según lo que espera tu modelo)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensiona la imagen, ajusta según tu modelo
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalización si es necesario
])

# Función para hacer la predicción
def predict_image(image):
    image = transform(image).unsqueeze(0)  # Agregar una dimensión extra para el batch
    with torch.no_grad():  # No calcular gradientes para la predicción
        outputs = model(image)  # Realizar la predicción
        _, predicted_class = torch.max(outputs, 1)  # Obtener la clase con mayor probabilidad
    return predicted_class.item()

# Crear la interfaz de la aplicación
st.title("Detección de Tumores Cerebrales")
st.write("Sube una imagen y el modelo hará la predicción.")

# Subir imagen
uploaded_file = st.file_uploader("Cargar Imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada.", use_column_width=True)

    # Realizar la predicción
    prediction = predict_image(image)

    if prediction == 0:

        prediction = 'Se detectó un Glioma'
        
    elif prediction == 1:

        prediction = ' Se detectó un Meningioma'

    elif prediction == 2:

        prediction = 'No se detectó ningún tumor'

    elif prediction == 3:

        prediction = ' Se detectó un tumor en la glándula pituitaria'
        
    st.write("{}".format(prediction))
