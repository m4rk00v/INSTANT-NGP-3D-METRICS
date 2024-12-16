# import numpy as np
# import open3d as o3d
# import cv2
# import torch
# from torchvision import transforms
# from torchvision.transforms import Compose,Resize,ToTensor,Normalize
# from scipy.spatial.distance import cdist
# from pyemd import emd
# import os
# from PIL import Image
# from torchvision.transforms import ToTensor


# import torch
# from PIL import Image
# from torchvision.transforms import Compose, ToTensor, Normalize, Resize
# import cv2



# # Paso 1: Cargar las imágenes normales
# def load_images(image_folder):
#     """
#     Carga las imágenes normales desde una carpeta.
#     """
#     image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     images = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in image_paths]
#     return images




# class DepthEstimator:
#     def __init__(self):
#         """
#         Carga el modelo MiDaS y define transformaciones personalizadas.
#         """
#         # Descargar el modelo MiDaS
#         self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
#         self.model.eval()

#         # Crear transformaciones personalizadas
#         self.transform = Compose([
#             Resize((384, 384)),  # Redimensionar la imagen al tamaño esperado
#             ToTensor(),  # Convertir a tensor
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizar
#         ])

#     def predict_depth(self, image):
#         """
#         Genera un mapa de profundidad para una imagen.
#         :param image: Imagen en formato numpy array (RGB).
#         :return: Mapa de profundidad (numpy array).
#         """
#         # Validación inicial: Verificar si la imagen tiene 3 canales (RGB)
#         if image is None or len(image.shape) != 3 or image.shape[2] != 3:
#             raise ValueError("La imagen debe tener 3 canales (RGB).")

#         # Convertir la imagen de BGR (OpenCV) a RGB
#         input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Convertir la imagen a un objeto PIL
#         input_image_pil = Image.fromarray(input_image)

#         # Aplicar las transformaciones personalizadas
#         transformed_image = self.transform(input_image_pil)  # Salida esperada: Tensor [C, H, W]

#         # Añadir una dimensión de batch (para convertir a [1, C, H, W])
#         input_tensor = transformed_image.unsqueeze(0)

#         # Validar dimensiones del tensor
#         print(f"Dimensiones del tensor antes del modelo: {input_tensor.shape}")
#         if input_tensor.ndimension() != 4 or input_tensor.shape[1] != 3:
#             raise ValueError(f"El tensor de entrada tiene dimensiones incorrectas: {input_tensor.shape}")

#         # Pasar por el modelo para generar el mapa de profundidad
#         with torch.no_grad():
#             prediction = self.model(input_tensor)  # Salida esperada: [1, H, W]
#             depth_map = prediction.squeeze().cpu().numpy()  # Convertir a numpy array

#         print(f"Tamaño del mapa de profundidad generado: {depth_map.shape}")
#         return depth_map


# # Paso 3: Convertir mapas de profundidad en nubes de puntos
# def depth_map_to_point_cloud(depth_map, fx, fy, cx, cy):
#     """
#     Convierte un mapa de profundidad en una nube de puntos.
#     """
#     h, w = depth_map.shape
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth_map

#     x = (x - cx) * z / fx
#     y = (y - cy) * z / fy

#     points = np.stack((x, y, z), axis=-1)
#     points = points.reshape(-1, 3)  # Convertir a (N, 3)
#     return points


# # Paso 4: Cargar el modelo 3D como nube de puntos
# def load_3d_model(file_path):
#     """
#     Carga el modelo 3D como una nube de puntos.
#     """
#     mesh = o3d.io.read_triangle_mesh(file_path)
#     pcd = mesh.sample_points_uniformly(number_of_points=5000)
#     print("array :",len(np.asarray(pcd.points)))
#     return np.asarray(pcd.points), pcd


# # Paso 5: Visualización de las nubes de puntos
# def visualize_point_clouds(model_cloud, depth_cloud):
#     """
#     Visualiza el modelo 3D y la nube de puntos generada a partir del depth map.
#     """
#     model_pcd = o3d.geometry.PointCloud()
#     model_pcd.points = o3d.utility.Vector3dVector(model_cloud)
#     model_pcd.paint_uniform_color([0, 0, 1])  # Azul para el modelo original

#     depth_pcd = o3d.geometry.PointCloud()
#     depth_pcd.points = o3d.utility.Vector3dVector(depth_cloud)
#     depth_pcd.paint_uniform_color([1, 0, 0])  # Rojo para la nube del depth map

#     # Combinar y mostrar
#     o3d.visualization.draw_geometries([model_pcd, depth_pcd], window_name="Modelo 3D y Depth Map")


# # Paso 6: Calcular EMD entre nubes de puntos
# def compute_emd(point_cloud_1, point_cloud_2):
#     """
#     Calcula la métrica EMD entre dos nubes de puntos.
#     """
#     distance_matrix = cdist(point_cloud_1, point_cloud_2, metric="euclidean")

#     # Asumir distribuciones uniformes para las dos nubes de puntos
#     weights1 = np.ones(len(point_cloud_1)) / len(point_cloud_1)
#     weights2 = np.ones(len(point_cloud_2)) / len(point_cloud_2)

#     # Convierte los pesos a listas para pyemd
#     weights1 = weights1.tolist()
#     weights2 = weights2.tolist()

#     emd_distance = emd(weights1, weights2, distance_matrix)
#     return emd_distance


# # Flujo principal
# if __name__ == "__main__":
  
#     # print(f"starting . . .")
#     # # Parámetros de cámara (supuestos para una cámara estándar)
#     # fx, fy = 500, 500  # Focal lengths
#     # cx, cy = 320, 240  # Centro óptico (suponer resolución 640x480)

#     # # Rutas de datos
#     # image_folder = "/home/kev/instant-ngp/data/unit/horns/images"  # Carpeta con imágenes normales
#     # model_3d_path = "/home/kev/Desktop/horns.obj"  # Archivo del modelo 3D

#     # # Cargar imágenes normales
#     # images = load_images(image_folder)

#     # # Inicializar estimador de profundidad
#     # depth_estimator = DepthEstimator()

#     # print(depth_estimator)

#     # # Cargar el modelo 3D como nube de puntos
#     # model_3d_cloud, model_pcd = load_3d_model(model_3d_path)

#     # # Procesar cada imagen
#     # for i, image in enumerate(images):
#     #     print(f"image nro : {i}")
#     #     # Generar mapa de profundidad y nube de puntos
#     #     depth_map = depth_estimator.predict_depth(image)
#     #     depth_cloud = depth_map_to_point_cloud(depth_map, fx, fy, cx, cy)

#     #     # Calcular EMD
#     #     emd_value = compute_emd(depth_cloud, model_3d_cloud)
#     #     print(f"EMD entre el mapa de profundidad {i + 1} y el modelo 3D: {emd_value:.4f}")

#     #     # Visualizar nubes de puntos
#     #     print(f"Visualizando el mapa de profundidad {i + 1} y el modelo 3D...")
#     #     visualize_point_clouds(model_3d_cloud, depth_cloud)




#     # Inicializar el estimador de profundidad
#     depth_estimator = DepthEstimator()

#     # Cargar una imagen desde un archivo
#     image_path = "/home/kev/instant-ngp/data/unit/horns/images/DJI_20200223_163016_842.jpg"
#     image = cv2.imread(image_path)

#     try:
#         # Generar el mapa de profundidad
#         depth_map = depth_estimator.predict_depth(image)

#         # Visualizar el mapa de profundidad
#         import matplotlib.pyplot 
#         matplotlib.use('Agg')

#         import matplotlib.pyplot as plt
#         plt.imshow(depth_map, cmap="plasma")
#         plt.colorbar()
#         plt.title("Mapa de Profundidad")
#         plt.savefig("/home/kev/instant-ngp/data/unit/horns/output_depthmap.jpg")

#     except Exception as e:
#         print(f"Error al generar el mapa de profundidad: {e}")



# import numpy as np
# import open3d as o3d
# import cv2
# import torch
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from scipy.spatial.distance import cdist
# from pyemd import emd
# import os
# from PIL import Image
# import matplotlib.pyplot as plt

# class DepthEstimator:
#     def __init__(self):
#         """
#         Carga el modelo MiDaS y define transformaciones personalizadas.
#         """
#         # Descargar el modelo MiDaS
#         self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
#         self.model.eval()

#         # Crear transformaciones personalizadas
#         self.transform = Compose([
#             Resize((384, 384)),  # Redimensionar la imagen al tamaño esperado
#             ToTensor(),  # Convertir a tensor
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizar
#         ])

#     def predict_depth(self, image):
#         """
#         Genera un mapa de profundidad para una imagen.
#         :param image: Imagen en formato numpy array (RGB).
#         :return: Mapa de profundidad (numpy array).
#         """
#         # Validar que la imagen tenga 3 canales
#         if image is None or len(image.shape) != 3 or image.shape[2] != 3:
#             raise ValueError("La imagen debe tener 3 canales (RGB).")

#         # Convertir de BGR a RGB
#         input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Convertir a PIL.Image
#         input_image_pil = Image.fromarray(input_image)

#         # Aplicar transformaciones
#         transformed_image = self.transform(input_image_pil)  # Salida esperada: Tensor [C, H, W]
#         input_tensor = transformed_image.unsqueeze(0)  # Añadir dimensión batch

#         # Generar el mapa de profundidad
#         with torch.no_grad():
#             prediction = self.model(input_tensor)  # Salida esperada: [1, H, W]
#             depth_map = prediction.squeeze().cpu().numpy()  # Convertir a numpy array

#         return depth_map

# def depth_map_to_point_cloud(depth_map, fx, fy, cx, cy):
#     """
#     Convierte un mapa de profundidad en una nube de puntos.
#     """
#     h, w = depth_map.shape
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth_map

#     x = (x - cx) * z / fx
#     y = (y - cy) * z / fy

#     points = np.stack((x, y, z), axis=-1)
#     points = points.reshape(-1, 3)  # Convertir a (N, 3)
#     return points

# def load_3d_model(file_path):
#     """
#     Carga el modelo 3D como una nube de puntos.
#     """
#     mesh = o3d.io.read_triangle_mesh(file_path)
#     pcd = mesh.sample_points_uniformly(number_of_points=5000)
#     return np.asarray(pcd.points), pcd

# def compute_emd(point_cloud_1, point_cloud_2):
#     """
#     Calcula la métrica EMD entre dos nubes de puntos después de igualar sus tamaños.
#     """
#     # Igualar los tamaños de las nubes de puntos
#     min_points = min(len(point_cloud_1), len(point_cloud_2))
#     point_cloud_1 = point_cloud_1[:min_points]
#     point_cloud_2 = point_cloud_2[:min_points]

#     # Calcular la matriz de distancia
#     distance_matrix = cdist(point_cloud_1, point_cloud_2, metric="euclidean")

#     # Asumir distribuciones uniformes para ambas nubes
#     weights1 = np.ones(len(point_cloud_1)) / len(point_cloud_1)
#     weights2 = np.ones(len(point_cloud_2)) / len(point_cloud_2)

#     # Calcular la métrica EMD
#     return emd(weights1, weights2, distance_matrix)

# def visualize_point_clouds(model_cloud, depth_cloud):
#     """
#     Visualiza el modelo 3D y la nube de puntos generada a partir del depth map.
#     """
#     model_pcd = o3d.geometry.PointCloud()
#     model_pcd.points = o3d.utility.Vector3dVector(model_cloud)
#     model_pcd.paint_uniform_color([0, 0, 1])  # Azul para el modelo original

#     depth_pcd = o3d.geometry.PointCloud()
#     depth_pcd.points = o3d.utility.Vector3dVector(depth_cloud)
#     depth_pcd.paint_uniform_color([1, 0, 0])  # Rojo para el depth map

#     # Combinar y mostrar
#     #o3d.visualization.draw_geometries([model_pcd, depth_pcd], window_name="Modelo 3D y Depth Map")
#     o3d.visualization.draw_geometries([model_pcd], window_name="Modelo 3D ")
#     o3d.visualization.draw_geometries([depth_pcd], window_name=" Depth Map")
#     o3d.visualization.draw_geometries([depth_pcd,model_pcd], window_name="Modelo 3D y Depth Map")

# def process_images(image_folder, model_3d_path, output_folder):
#     """
#     Procesa las imágenes para generar mapas de profundidad y calcular EMD.
#     """
#     # Parámetros de cámara
#     fx, fy = 500, 500  # Longitudes focales
#     cx, cy = 320, 240  # Centro óptico (asumiendo resolución 640x480)

#     # Inicializar el estimador de profundidad
#     depth_estimator = DepthEstimator()

#     # Cargar el modelo 3D
#     model_3d_cloud, _ = load_3d_model(model_3d_path)

#     # Procesar imágenes en la carpeta
#     image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

#     for i, image_path in enumerate(image_paths):
#         image = cv2.imread(image_path)

#         # Generar mapa de profundidad
#         depth_map = depth_estimator.predict_depth(image)

#         # Convertir el mapa de profundidad en nube de puntos
#         depth_cloud = depth_map_to_point_cloud(depth_map, fx, fy, cx, cy)

#         # Calcular la métrica EMD
#         emd_value = compute_emd(depth_cloud, model_3d_cloud)
#         print(f"EMD for {os.path.basename(image_path)}: {emd_value:.4f}")

#         # Guardar el mapa de profundidad
#         # plt.imshow(depth_map, cmap="plasma")
#         # plt.colorbar()
#         # plt.title(f"Depth Map: {os.path.basename(image_path)}")
#         # plt.savefig(os.path.join(output_folder, f"depth_map_{i + 1}.png"))
#         # plt.close()

#         # Visualización de las nubes de puntos
#         visualize_point_clouds(model_3d_cloud, depth_cloud)

# if __name__ == "__main__":
#     # Carpetas de entrada y salida
#     image_folder = "/home/kev/instant-ngp/data/unit/horns/images"
#     model_3d_path = "/home/kev/Desktop/horns.obj"
#     output_folder = "/home/kev/instant-ngp/data/unit/horns/output"

#     os.makedirs(output_folder, exist_ok=True)

#     # Procesar las imágenes y calcular EMD
#     process_images(image_folder, model_3d_path, output_folder)



# import numpy as np
# import open3d as o3d
# import cv2
# import torch
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from scipy.spatial.distance import cdist
# from pyemd import emd
# import os
# from PIL import Image
# import matplotlib.pyplot 
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


# class DepthEstimator:
#     def __init__(self):
#         """
#         Load the MiDaS model and define custom transformations.
#         """
#         self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
#         self.model.eval()

#         # Custom transformations
#         self.transform = Compose([
#             Resize((384, 384)),  # Resize the input to match model requirements
#             ToTensor(),  # Convert to tensor
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
#         ])

#     def predict_depth(self, image):
#         """
#         Generate a depth map for an input image.
#         :param image: Input image as a numpy array (RGB).
#         :return: Depth map as a numpy array.
#         """
#         if image is None or len(image.shape) != 3 or image.shape[2] != 3:
#             raise ValueError("The input image must have 3 channels (RGB).")

#         # Convert from BGR to RGB
#         input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Convert to PIL Image
#         input_image_pil = Image.fromarray(input_image)

#         # Apply transformations
#         transformed_image = self.transform(input_image_pil)
#         input_tensor = transformed_image.unsqueeze(0)  # Add batch dimension

#         # Generate the depth map
#         with torch.no_grad():
#             prediction = self.model(input_tensor)
#             depth_map = prediction.squeeze().cpu().numpy()

#         return depth_map


# def depth_map_to_point_cloud(depth_map, fl_x, fl_y, cx, cy):
#     """
#     Convert a depth map to a 3D point cloud using camera parameters.
#     """
#     h, w = depth_map.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth_map

#     x = (i - cx) * z / fl_x
#     y = (j - cy) * z / fl_y

#     points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     return points


# def normalize_point_cloud(points):
#     """
#     Normalize a point cloud to center it at the origin and scale it uniformly.
#     """
#     centroid = np.mean(points, axis=0)
#     points -= centroid
#     max_distance = np.max(np.linalg.norm(points, axis=1))
#     points /= max_distance
#     return points


# def load_3d_model(file_path):
#     """
#     Load a 3D model as a point cloud.
#     """
#     mesh = o3d.io.read_triangle_mesh(file_path)
#     pcd = mesh.sample_points_uniformly(number_of_points=5000)
#     return np.asarray(pcd.points), pcd


# def compute_emd(point_cloud_1, point_cloud_2):
#     """
#     Compute the Earth Mover's Distance (EMD) between two point clouds.
#     """
#     # Match the size of the two point clouds
#     min_points = min(len(point_cloud_1), len(point_cloud_2))
#     point_cloud_1 = point_cloud_1[:min_points]
#     point_cloud_2 = point_cloud_2[:min_points]

#     distance_matrix = cdist(point_cloud_1, point_cloud_2, metric="euclidean")
#     weights1 = np.ones(len(point_cloud_1)) / len(point_cloud_1)
#     weights2 = np.ones(len(point_cloud_2)) / len(point_cloud_2)

#     return emd(weights1, weights2, distance_matrix)


# def visualize_single_cloud(point_cloud, color):
#     """
#     Visualize a single point cloud.
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud)
#     pcd.paint_uniform_color(color)
#     o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")


# def visualize_combined_clouds(model_cloud, depth_cloud):
#     """
#     Visualize the 3D model and depth map point clouds together.
#     """
#     model_pcd = o3d.geometry.PointCloud()
#     model_pcd.points = o3d.utility.Vector3dVector(model_cloud)
#     model_pcd.paint_uniform_color([0, 0, 1])  # Blue for the model

#     depth_pcd = o3d.geometry.PointCloud()
#     depth_pcd.points = o3d.utility.Vector3dVector(depth_cloud)
#     depth_pcd.paint_uniform_color([1, 0, 0])  # Red for the depth map

#     o3d.visualization.draw_geometries([model_pcd, depth_pcd], window_name="3D Model and Depth Map")


# def process_images(image_folder, model_3d_path, output_folder, camera_params):
#     """
#     Process images to generate depth maps, compute EMD, and visualize results.
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     depth_estimator = DepthEstimator()
#     model_3d_cloud, _ = load_3d_model(model_3d_path)
#     model_3d_cloud = normalize_point_cloud(model_3d_cloud)

#     image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

#     for i, image_path in enumerate(image_paths):
#         image = cv2.imread(image_path)

#         # Generate depth map
#         depth_map = depth_estimator.predict_depth(image)

#         # Convert depth map to point cloud
#         depth_cloud = depth_map_to_point_cloud(
#             depth_map,
#             fl_x=camera_params["fl_x"],
#             fl_y=camera_params["fl_y"],
#             cx=camera_params["cx"],
#             cy=camera_params["cy"]
#         )
#         depth_cloud = normalize_point_cloud(depth_cloud)

#         # Compute EMD
        
#         print(f"Model cloud points: {model_3d_cloud.shape}, Depth map points: {depth_cloud.shape}")
#         print(f"Model cloud range: x({model_3d_cloud[:, 0].min()}, {model_3d_cloud[:, 0].max()}), "f"y({model_3d_cloud[:, 1].min()}, {model_3d_cloud[:, 1].max()}), "f"z({model_3d_cloud[:, 2].min()}, {model_3d_cloud[:, 2].max()})")
#         print(f"Depth map cloud range: x({depth_cloud[:, 0].min()}, {depth_cloud[:, 0].max()}), "f"y({depth_cloud[:, 1].min()}, {depth_cloud[:, 1].max()}), "f"z({depth_cloud[:, 2].min()}, {depth_cloud[:, 2].max()})")
#         emd_value = compute_emd(depth_cloud, model_3d_cloud)
#         print(f"EMD for {os.path.basename(image_path)}: {emd_value:.4f}")

#         # Save depth map visualization
#         plt.imshow(depth_map, cmap="plasma")
#         plt.colorbar()
#         plt.title(f"Depth Map: {os.path.basename(image_path)}")
#         plt.savefig(os.path.join(output_folder, f"depth_map_{i + 1}.png"))
#         plt.close()

#         # Visualize combined point clouds
#         visualize_combined_clouds(model_3d_cloud, depth_cloud)


# if __name__ == "__main__":
#     camera_params = {
#         "fl_x": 500,
#         "fl_y": 500,
#         "cx": 320,
#         "cy": 240,
#         "w": 4032.0,
#         "h": 3024.0
#     }

#      # # Parámetros de cámara (supuestos para una cámara estándar)
# #     # fx, fy = 500, 500  # Focal lengths
# #     # cx, cy = 320, 240  # Centro óptico (suponer resolución 640x480)

#     image_folder = "/home/kev/instant-ngp/data/unit/horns/images"
#     model_3d_path = "/home/kev/Desktop/horns.obj"
#     output_folder = "/home/kev/instant-ngp/data/unit/horns/output"
#     process_images(image_folder, model_3d_path, output_folder, camera_params)




# import numpy as np
# import open3d as o3d
# import cv2
# import torch
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from scipy.spatial.distance import cdist
# from pyemd import emd
# import os
# from PIL import Image
# # import matplotlib.pyplot
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from scipy.spatial.distance import cdist
# import numpy as np
# import ot  # Python Optimal Transport


# image_folder = "/home/kev/instant-ngp/data/unit/horns/images"
# model_3d_path = "/home/kev/Desktop/horns.obj"
# # model_3d_path = "/home/kev/instant-ngp/data/nerf_llff_data/meshs_output/fortress/fortress.obj"
# output_folder = "/home/kev/instant-ngp/data/unit/horns/output/"

# class DepthEstimator:
#     def __init__(self):
#         """
#         Load the MiDaS model and define custom transformations.
#         """
#         self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
#         self.model.eval()

#         # Custom transformations
#         self.transform = Compose([
#             Resize((384, 384)),  # Resize the input to match model requirements
#             ToTensor(),  # Convert to tensor
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
#         ])

#     def predict_depth(self, image):
#         """
#         Generate a depth map for an input image.
#         :param image: Input image as a numpy array (RGB).
#         :return: Depth map as a numpy array.
#         """
#         if image is None or len(image.shape) != 3 or image.shape[2] != 3:
#             raise ValueError("The input image must have 3 channels (RGB).")

#         # Convert from BGR to RGB
#         input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Convert to PIL Image
#         input_image_pil = Image.fromarray(input_image)

#         # Apply transformations
#         transformed_image = self.transform(input_image_pil)
#         input_tensor = transformed_image.unsqueeze(0)  # Add batch dimension

#         # Generate the depth map
#         with torch.no_grad():
#             prediction = self.model(input_tensor)
#             depth_map = prediction.squeeze().cpu().numpy()

#         return depth_map

# def depth_map_to_point_cloud(depth_map, fl_x, fl_y, cx, cy):
#     """
#     Convert a depth map to a 3D point cloud using camera parameters.
#     """
#     h, w = depth_map.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth_map

#     x = (i - cx) * z / fl_x
#     y = (j - cy) * z / fl_y

#     points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     return points

# def normalize_point_cloud(points):
#     """
#     Normalize a point cloud to center it at the origin and scale it uniformly.
#     """
#     centroid = np.mean(points, axis=0)
#     points -= centroid
#     max_distance = np.max(np.linalg.norm(points, axis=1))
#     points /= max_distance
#     return points

# def sample_points(point_cloud, num_points):
#     """
#     Reduce the number of points in a point cloud to a fixed amount (num_points).
#     If the point cloud has fewer points than num_points, it is returned as is.
#     :param point_cloud: Input point cloud as a numpy array of shape (N, 3).
#     :param num_points: Number of points to sample.
#     :return: Downsampled point cloud with shape (num_points, 3).
#     """
#     if len(point_cloud) <= num_points:
#         return point_cloud  # If already smaller or equal, return as is
#     indices = np.random.choice(len(point_cloud), num_points, replace=False)  # Random sampling
#     return point_cloud[indices]

# def load_3d_model(file_path):
#     """
#     Load a 3D model as a point cloud.
#     """
#     mesh = o3d.io.read_triangle_mesh(file_path)
#     pcd = mesh.sample_points_uniformly(number_of_points=5000)
#     return np.asarray(pcd.points), pcd



# def compute_emd(depth_point_cloud, threed_point_cloud,i):
#     """
#     Compute the Earth Mover's Distance (EMD) between two point clouds
#     using the POT (Python Optimal Transport) library.
#     """
#     # Igualar el tamaño de los point clouds
#     min_points = min(len(depth_point_cloud), len(threed_point_cloud))
#     depth_point_cloud = depth_point_cloud[:min_points]
#     threed_point_cloud = threed_point_cloud[:min_points]

#     plot_point_clouds(depth_point_cloud, threed_point_cloud,i)
#     # euclidean distance calculation
#     distance_matrix = cdist(depth_point_cloud, threed_point_cloud, metric='euclidean')

#     # weight normalizated distribution
#     weights1 = np.ones(len(depth_point_cloud)) / len(depth_point_cloud)
#     weights2 = np.ones(len(threed_point_cloud)) / len(threed_point_cloud)

#     # EMD calculation using  optimize transport
#     emd_value = ot.emd2(weights1, weights2, distance_matrix)

#     return emd_value



# def plot_point_clouds(pc1, pc2,i):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], label='depth cloud points', alpha=0.5)
#     ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], label='3D model cloud points ', alpha=0.5)
#     ax.set_title("Cloud points comparison")
#     plt.legend()
#     plt.savefig(f"/home/kev/instant-ngp/data/unit/horns/output/combined_{i}")
#     plt.close()

# def visualize_combined_clouds(model_cloud, depth_cloud,i):
#     """
#     Visualize the 3D model and depth map point clouds together.
#     """
#     model_pcd = o3d.geometry.PointCloud()
#     model_pcd.points = o3d.utility.Vector3dVector(model_cloud)
#     model_pcd.paint_uniform_color([0, 0, 1])  # Blue for the model

#     depth_pcd = o3d.geometry.PointCloud()
#     depth_pcd.points = o3d.utility.Vector3dVector(depth_cloud)
#     depth_pcd.paint_uniform_color([1, 0, 0])  # Red for the depth map

#     #o3d.visualization.draw_geometries([model_pcd, depth_pcd], window_name="3D Model and Depth Map")
#     # o3d.visualization.draw_geometries([model_pcd], window_name="3D Model")
#     # o3d.visualization.draw_geometries([depth_pcd], window_name="Depth Map")
   

# def process_images(image_folder, model_3d_path, output_folder, camera_params):
#     """
#     Process images to generate depth maps, compute EMD, and visualize results.
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     depth_estimator = DepthEstimator()

#     # Load the 3D model and normalize it
#     model_3d_cloud, _ = load_3d_model(model_3d_path)
#     model_3d_cloud = normalize_point_cloud(model_3d_cloud)
#     model_3d_cloud = sample_points(model_3d_cloud, 5000)  # Ensure it has 5000 points

#     image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

#     for i, image_path in enumerate(image_paths):
#         image = cv2.imread(image_path)

#         # Generate depth map
#         depth_map = depth_estimator.predict_depth(image)

#         # Convert depth map to point cloud
#         depth_cloud = depth_map_to_point_cloud(
#             depth_map,
#             fl_x=camera_params["fl_x"],
#             fl_y=camera_params["fl_y"],
#             cx=camera_params["cx"],
#             cy=camera_params["cy"]
#         )
#         depth_cloud = normalize_point_cloud(depth_cloud)
#         depth_cloud = sample_points(depth_cloud, 10000)  # Match the model 3D points

#         # Compute EMD
#         emd_value = compute_emd(depth_cloud, model_3d_cloud,i)

#         print(f"Model cloud points: {model_3d_cloud.shape}, Depth map points: {depth_cloud.shape}")
#         print(f"Model cloud range: x({model_3d_cloud[:, 0].min()}, {model_3d_cloud[:, 0].max()}), "
#               f"y({model_3d_cloud[:, 1].min()}, {model_3d_cloud[:, 1].max()}), "
#               f"z({model_3d_cloud[:, 2].min()}, {model_3d_cloud[:, 2].max()})")
#         print(f"Depth map cloud range: x({depth_cloud[:, 0].min()}, {depth_cloud[:, 0].max()}), "
#               f"y({depth_cloud[:, 1].min()}, {depth_cloud[:, 1].max()}), "
#               f"z({depth_cloud[:, 2].min()}, {depth_cloud[:, 2].max()})")
        
#         print(f"EMD for {os.path.basename(image_path)}: {emd_value:.9f}")

#         # Save depth map visualization
#         plt.imshow(depth_map, cmap="plasma")
#         plt.colorbar()
#         plt.title(f"Depth Map: {os.path.basename(image_path)}")
#         plt.savefig(os.path.join(output_folder, f"depth_map_{i + 1}.png"))
#         # plt.show()
#         plt.close()

#         # Visualize combined point clouds
#         visualize_combined_clouds(model_3d_cloud, depth_cloud,i)

# if __name__ == "__main__":
#     camera_params = {
#             "fl_x": 500,
#             "fl_y": 500,
#             "cx": 320,
#             "cy": 240,
#             "w": 4032.0,
#             "h": 4032.0
#         }

 
#     process_images(image_folder, model_3d_path, output_folder, camera_params)


# import numpy as np
# import open3d as o3d
# import cv2
# import torch
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from scipy.spatial.distance import cdist
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import ot  # Python Optimal Transport
# import csv

# image_folder = "/home/kev/instant-ngp/data/unit/horns/images"
# model_3d_path = "/home/kev/Desktop/horns.obj"
# output_folder = "/home/kev/instant-ngp/data/unit/horns/output/"

# class DepthEstimator:
#     def __init__(self):
#         """
#         Load the MiDaS model and define custom transformations.
#         """
#         print("Loading MiDaS model...")
#         self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
#         self.model.eval()
#         print("MiDaS model loaded.")

#         # Custom transformations
#         self.transform = Compose([
#             Resize((384, 384)),  # Resize the input to match model requirements
#             ToTensor(),  # Convert to tensor
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
#         ])

#     def predict_depth(self, image):
#         """
#         Generate a depth map for an input image.
#         :param image: Input image as a numpy array (RGB).
#         :return: Depth map as a numpy array.
#         """
#         if image is None or len(image.shape) != 3 or image.shape[2] != 3:
#             raise ValueError("The input image must have 3 channels (RGB).")

#         print("Generating depth map...")

#         # Convert from BGR to RGB
#         input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Convert to PIL Image
#         input_image_pil = Image.fromarray(input_image)

#         # Apply transformations
#         transformed_image = self.transform(input_image_pil)
#         input_tensor = transformed_image.unsqueeze(0)  # Add batch dimension

#         # Generate the depth map
#         with torch.no_grad():
#             prediction = self.model(input_tensor)
#             depth_map = prediction.squeeze().cpu().numpy()

#         print("Depth map generated.")
#         return depth_map

# def depth_map_to_point_cloud(depth_map, fl_x, fl_y, cx, cy):
#     """
#     Convert a depth map to a 3D point cloud using camera parameters.
#     """
#     print("Converting depth map to point cloud...")
#     h, w = depth_map.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth_map

#     x = (i - cx) * z / fl_x
#     y = (j - cy) * z / fl_y

#     points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     print("Point cloud conversion complete.")
#     return points

# def normalize_point_cloud(points):
#     """
#     Normalize a point cloud to center it at the origin and scale it uniformly.
#     """
#     print("Normalizing point cloud...")
#     centroid = np.mean(points, axis=0)
#     points -= centroid
#     max_distance = np.max(np.linalg.norm(points, axis=1))
#     points /= max_distance
#     print("Point cloud normalization complete.")
#     return points

# def sample_points(point_cloud, num_points):
#     """
#     Reduce the number of points in a point cloud to a fixed amount (num_points).
#     If the point cloud has fewer points than num_points, it is returned as is.
#     :param point_cloud: Input point cloud as a numpy array of shape (N, 3).
#     :param num_points: Number of points to sample.
#     :return: Downsampled point cloud with shape (num_points, 3).
#     """
#     print("Sampling point cloud...")
#     if len(point_cloud) <= num_points:
#         return point_cloud  # If already smaller or equal, return as is
#     indices = np.random.choice(len(point_cloud), num_points, replace=False)  # Random sampling
#     print("Point cloud sampling complete.")
#     return point_cloud[indices]

# def load_3d_model(file_path):
#     """
#     Load a 3D model as a point cloud.
#     """
#     print("Loading 3D model...")
#     mesh = o3d.io.read_triangle_mesh(file_path)
#     pcd = mesh.sample_points_uniformly(number_of_points=5000)
#     print("3D model loaded.")
#     return np.asarray(pcd.points), pcd

# def compute_emd(depth_point_cloud, threed_point_cloud, i):
#     """
#     Compute the Earth Mover's Distance (EMD) between two point clouds
#     using the POT (Python Optimal Transport) library.
#     """
#     print(f"Computing EMD for image {i + 1}...")

#     # Equalize the size of the point clouds
#     min_points = min(len(depth_point_cloud), len(threed_point_cloud))
#     depth_point_cloud = depth_point_cloud[:min_points]
#     threed_point_cloud = threed_point_cloud[:min_points]

#     plot_point_clouds(depth_point_cloud, threed_point_cloud, i)

#     # Euclidean distance calculation
#     distance_matrix = cdist(depth_point_cloud, threed_point_cloud, metric='euclidean')

#     # Weight normalized distribution
#     weights1 = np.ones(len(depth_point_cloud)) / len(depth_point_cloud)
#     weights2 = np.ones(len(threed_point_cloud)) / len(threed_point_cloud)

#     # EMD calculation using optimal transport
#     emd_value = ot.emd2(weights1, weights2, distance_matrix)

#     print(f"EMD for image {i + 1} computed: {emd_value:.9f}")
#     return emd_value

# def plot_point_clouds(pc1, pc2, i):
#     print(f"Plotting point clouds for image {i + 1}...")
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], label='Depth cloud points', alpha=0.5)
#     ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], label='3D model cloud points', alpha=0.5)
#     ax.set_title("Cloud points comparison")
#     plt.legend()
#     plt.savefig(f"/home/kev/instant-ngp/data/unit/horns/output/combined_{i}")
#     plt.close()
#     print(f"Point clouds for image {i + 1} plotted and saved.")

# def visualize_combined_clouds(model_cloud, depth_cloud, i):
#     """
#     Visualize the 3D model and depth map point clouds together.
#     """
#     print(f"Visualizing combined point clouds for image {i + 1}...")

#     model_pcd = o3d.geometry.PointCloud()
#     model_pcd.points = o3d.utility.Vector3dVector(model_cloud)
#     model_pcd.paint_uniform_color([0, 0, 1])  # Blue for the model

#     depth_pcd = o3d.geometry.PointCloud()
#     depth_pcd.points = o3d.utility.Vector3dVector(depth_cloud)
#     depth_pcd.paint_uniform_color([1, 0, 0])  # Red for the depth map

#     print(f"Visualization for image {i + 1} complete.")

# def process_images(image_folder, model_3d_path, output_folder, camera_params):
#     """
#     Process images to generate depth maps, compute EMD, and visualize results.
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     depth_estimator = DepthEstimator()

#     # Load the 3D model and normalize it
#     model_3d_cloud, _ = load_3d_model(model_3d_path)
#     model_3d_cloud = normalize_point_cloud(model_3d_cloud)
#     model_3d_cloud = sample_points(model_3d_cloud, 5000)  # Ensure it has 5000 points

#     image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

#     emd_values = []

#     for i, image_path in enumerate(image_paths):
#         print(f"Processing image {i + 1}/{len(image_paths)}: {image_path}...")
#         image = cv2.imread(image_path)

#         # Generate depth map
#         depth_map = depth_estimator.predict_depth(image)

#         # Convert depth map to point cloud
#         depth_cloud = depth_map_to_point_cloud(
#             depth_map,
#             fl_x=camera_params["fl_x"],
#             fl_y=camera_params["fl_y"],
#             cx=camera_params["cx"],
#             cy=camera_params["cy"]
#         )
#         depth_cloud = normalize_point_cloud(depth_cloud)
#         depth_cloud = sample_points(depth_cloud, 10000)  # Match the model 3D points

#         # Compute EMD
#         emd_value = compute_emd(depth_cloud, model_3d_cloud, i)
#         emd_values.append((os.path.basename(image_path), emd_value))

#         # Save depth map visualization
#         plt.imshow(depth_map, cmap="plasma")
#         plt.colorbar()
#         plt.title(f"Depth Map: {os.path.basename(image_path)}")
#         plt.savefig(os.path.join(output_folder, f"depth_map_{i + 1}.png"))
#         plt.close()

#         # Visualize combined point clouds
#         visualize_combined_clouds(model_3d_cloud, depth_cloud, i)

#     # Save EMD values to CSV
#     csv_path = os.path.join(output_folder, "emd_values.csv")
#     print(f"Saving EMD values to {csv_path}...")
#     with open(csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Image", "EMD Value"])
#         writer.writerows(emd_values)

#     # Compute and print average EMD
#     average_emd = np.mean([value[1] for value in emd_values])
#     print(f"Average EMD: {average_emd:.9f}")
#     with open(csv_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([])
#         writer.writerow(["Average EMD", average_emd])

# if __name__ == "__main__":
#     camera_params = {
#             "fl_x": 500,
#             "fl_y": 500,
#             "cx": 320,
#             "cy": 240,
#             "w": 4032.0,
#             "h": 4032.0
#         }

#     process_images(image_folder, model_3d_path, output_folder, camera_params)


import numpy as np
import open3d as o3d
import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from scipy.spatial.distance import cdist
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ot  # Python Optimal Transport
import csv

# import numpy as np
# import open3d as o3d
# import cv2
# import torch
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from scipy.spatial.distance import cdist
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import ot  # Python Optimal Transport
# import csv

# image_folder = "/home/kev/instant-ngp/data/unit/horns/images"
# model_3d_path = "/home/kev/Desktop/horns.obj"
# output_folder = "/home/kev/instant-ngp/data/unit/horns/output/"

# class DepthEstimator:
#     def __init__(self):
#         """
#         Load the MiDaS model and define custom transformations.
#         """
#         print("Loading MiDaS model...")
#         self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
#         self.model.eval()
#         print("MiDaS model loaded.")

#         # Custom transformations
#         self.transform = Compose([
#             Resize((384, 384)),  # Resize the input to match model requirements
#             ToTensor(),  # Convert to tensor
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
#         ])

#     def predict_depth(self, image):
#         """
#         Generate a depth map for an input image.
#         :param image: Input image as a numpy array (RGB).
#         :return: Depth map as a numpy array.
#         """
#         if image is None or len(image.shape) != 3 or image.shape[2] != 3:
#             raise ValueError("The input image must have 3 channels (RGB).")

#         print("Generating depth map...")

#         # Convert from BGR to RGB
#         input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Convert to PIL Image
#         input_image_pil = Image.fromarray(input_image)

#         # Apply transformations
#         transformed_image = self.transform(input_image_pil)
#         input_tensor = transformed_image.unsqueeze(0)  # Add batch dimension

#         # Generate the depth map
#         with torch.no_grad():
#             prediction = self.model(input_tensor)
#             depth_map = prediction.squeeze().cpu().numpy()

#         print("Depth map generated.")
#         return depth_map

# def depth_map_to_point_cloud(depth_map, fl_x, fl_y, cx, cy):
#     """
#     Convert a depth map to a 3D point cloud using camera parameters.
#     """
#     print("Converting depth map to point cloud...")
#     h, w = depth_map.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth_map

#     x = (i - cx) * z / fl_x
#     y = (j - cy) * z / fl_y

#     points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     print("Point cloud conversion complete.")
#     return points

# def normalize_point_cloud(points):
#     """
#     Normalize a point cloud to center it at the origin and scale it uniformly.
#     """
#     print("Normalizing point cloud...")
#     centroid = np.mean(points, axis=0)
#     points -= centroid
#     max_distance = np.max(np.linalg.norm(points, axis=1))
#     points /= max_distance
#     print("Point cloud normalization complete.")
#     return points

# def sample_points(point_cloud, num_points):
#     """
#     Reduce the number of points in a point cloud to a fixed amount (num_points).
#     If the point cloud has fewer points than num_points, it is returned as is.
#     :param point_cloud: Input point cloud as a numpy array of shape (N, 3).
#     :param num_points: Number of points to sample.
#     :return: Downsampled point cloud with shape (num_points, 3).
#     """
#     print("Sampling point cloud...")
#     if len(point_cloud) <= num_points:
#         return point_cloud  # If already smaller or equal, return as is
#     indices = np.random.choice(len(point_cloud), num_points, replace=False)  # Random sampling
#     print("Point cloud sampling complete.")
#     return point_cloud[indices]

# def load_3d_model(file_path):
#     """
#     Load a 3D model as a point cloud.
#     """
#     print("Loading 3D model...")
#     mesh = o3d.io.read_triangle_mesh(file_path)
#     pcd = mesh.sample_points_uniformly(number_of_points=5000)
#     print("3D model loaded.")
#     return np.asarray(pcd.points), pcd

# def compute_emd(depth_point_cloud, threed_point_cloud, i):
#     """
#     Compute the Earth Mover's Distance (EMD) between two point clouds
#     using the POT (Python Optimal Transport) library.
#     """
#     print(f"Computing EMD for image {i + 1}...")

#     # Equalize the size of the point clouds
#     min_points = min(len(depth_point_cloud), len(threed_point_cloud))
#     depth_point_cloud = depth_point_cloud[:min_points]
#     threed_point_cloud = threed_point_cloud[:min_points]

#     plot_point_clouds(depth_point_cloud, threed_point_cloud, i)

#     # Euclidean distance calculation
#     distance_matrix = cdist(depth_point_cloud, threed_point_cloud, metric='euclidean')

#     # Weight normalized distribution
#     weights1 = np.ones(len(depth_point_cloud)) / len(depth_point_cloud)
#     weights2 = np.ones(len(threed_point_cloud)) / len(threed_point_cloud)

#     # EMD calculation using optimal transport
#     emd_value = ot.emd2(weights1, weights2, distance_matrix)

#     print(f"EMD for image {i + 1} computed: {emd_value:.9f}")
#     return emd_value

# def plot_point_clouds(pc1, pc2, i):
#     print(f"Plotting point clouds for image {i + 1}...")
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], label='Depth cloud points', alpha=0.5)
#     ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], label='3D model cloud points', alpha=0.5)
#     ax.set_title("Cloud points comparison")
#     plt.legend()
#     plt.savefig(f"/home/kev/instant-ngp/data/unit/horns/output/combined_{i}")
#     plt.close()
#     print(f"Point clouds for image {i + 1} plotted and saved.")

# def visualize_combined_clouds(model_cloud, depth_cloud, i):
#     """
#     Visualize the 3D model and depth map point clouds together.
#     """
#     print(f"Visualizing combined point clouds for image {i + 1}...")

#     model_pcd = o3d.geometry.PointCloud()
#     model_pcd.points = o3d.utility.Vector3dVector(model_cloud)
#     model_pcd.paint_uniform_color([0, 0, 1])  # Blue for the model

#     depth_pcd = o3d.geometry.PointCloud()
#     depth_pcd.points = o3d.utility.Vector3dVector(depth_cloud)
#     depth_pcd.paint_uniform_color([1, 0, 0])  # Red for the depth map

#     print(f"Visualization for image {i + 1} complete.")

# def process_images(image_folder, model_3d_path, output_folder, camera_params):
#     """
#     Process images to generate depth maps, compute EMD, and visualize results.
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     depth_estimator = DepthEstimator()

#     # Load the 3D model and normalize it
#     model_3d_cloud, _ = load_3d_model(model_3d_path)
#     model_3d_cloud = normalize_point_cloud(model_3d_cloud)
#     model_3d_cloud = sample_points(model_3d_cloud, 5000)  # Ensure it has 5000 points

#     image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

#     emd_values = []

#     for i, image_path in enumerate(image_paths):
#         print(f"Processing image {i + 1}/{len(image_paths)}: {image_path}...")
#         image = cv2.imread(image_path)

#         # Generate depth map
#         depth_map = depth_estimator.predict_depth(image)

#         # Convert depth map to point cloud
#         depth_cloud = depth_map_to_point_cloud(
#             depth_map,
#             fl_x=camera_params["fl_x"],
#             fl_y=camera_params["fl_y"],
#             cx=camera_params["cx"],
#             cy=camera_params["cy"]
#         )
#         depth_cloud = normalize_point_cloud(depth_cloud)
#         depth_cloud = sample_points(depth_cloud, 10000)  # Match the model 3D points

#         # Compute EMD
#         emd_value = compute_emd(depth_cloud, model_3d_cloud, i)
#         emd_values.append((os.path.basename(image_path), emd_value))

#         # Save depth map visualization
#         plt.imshow(depth_map, cmap="plasma")
#         plt.colorbar()
#         plt.title(f"Depth Map: {os.path.basename(image_path)}")
#         plt.savefig(os.path.join(output_folder, f"depth_map_{i + 1}.png"))
#         plt.close()

#         # Visualize combined point clouds
#         visualize_combined_clouds(model_3d_cloud, depth_cloud, i)

#     # Save EMD values to CSV
#     csv_path = os.path.join(output_folder, "emd_values.csv")
#     print(f"Saving EMD values to {csv_path}...")
#     with open(csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Image", "EMD Value"])
#         writer.writerows(emd_values)

#     # Compute and print average EMD
#     average_emd = np.mean([value[1] for value in emd_values])
#     print(f"Average EMD: {average_emd:.9f}")
#     with open(csv_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([])
#         writer.writerow(["Average EMD", average_emd])

# if __name__ == "__main__":
#     camera_params = {
#             "fl_x": 500,
#             "fl_y": 500,
#             "cx": 320,
#             "cy": 240,
#             "w": 4032.0,
#             "h": 4032.0
#         }

#     process_images(image_folder, model_3d_path, output_folder, camera_params)
root_folder = "/home/kev/instant-ngp/data/unit/"
output_root_folder = "/home/kev/instant-ngp/data/unit/output/"


# root_folder = "/home/kev/instant-ngp/data/unit_attacked/"
# output_root_folder = "/home/kev/instant-ngp/data/unit_attacked/output/"

class DepthEstimator:
    def __init__(self):
        """
        Load the MiDaS model and define custom transformations.
        """
        print("Loading MiDaS model...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.model.eval()
        print("MiDaS model loaded.")

        # Custom transformations
        self.transform = Compose([
            Resize((384, 384)),  # Resize the input to match model requirements
            ToTensor(),  # Convert to tensor
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

    def predict_depth(self, image):
        """
        Generate a depth map for an input image.
        :param image: Input image as a numpy array (RGB).
        :return: Depth map as a numpy array.
        """
        if image is None or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("The input image must have 3 channels (RGB).")

        print("Generating depth map...")

        # Convert from BGR to RGB
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        input_image_pil = Image.fromarray(input_image)

        # Apply transformations
        transformed_image = self.transform(input_image_pil)
        input_tensor = transformed_image.unsqueeze(0)  # Add batch dimension

        # Generate the depth map
        with torch.no_grad():
            prediction = self.model(input_tensor)
            depth_map = prediction.squeeze().cpu().numpy()

        print("Depth map generated.")
        return depth_map

def depth_map_to_point_cloud(depth_map, fl_x, fl_y, cx, cy):
    """
    Convert a depth map to a 3D point cloud using camera parameters.
    """
    print("Converting depth map to point cloud...")
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map

    x = (i - cx) * z / fl_x
    y = (j - cy) * z / fl_y

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    print("Point cloud conversion complete.")
    return points

def normalize_point_cloud(points):
    """
    Normalize a point cloud to center it at the origin and scale it uniformly.
    """
    print("Normalizing point cloud...")
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_distance = np.max(np.linalg.norm(points, axis=1))
    points /= max_distance
    print("Point cloud normalization complete.")
    return points

def sample_points(point_cloud, num_points):
    """
    Reduce the number of points in a point cloud to a fixed amount (num_points).
    If the point cloud has fewer points than num_points, it is returned as is.
    :param point_cloud: Input point cloud as a numpy array of shape (N, 3).
    :param num_points: Number of points to sample.
    :return: Downsampled point cloud with shape (num_points, 3).
    """
    print("Sampling point cloud...")
    if len(point_cloud) <= num_points:
        return point_cloud  # If already smaller or equal, return as is
    indices = np.random.choice(len(point_cloud), num_points, replace=False)  # Random sampling
    print("Point cloud sampling complete.")
    return point_cloud[indices]

def load_3d_model(file_path):
    """
    Load a 3D model as a point cloud.
    """
    print("Loading 3D model...")
    mesh = o3d.io.read_triangle_mesh(file_path)
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    print("3D model loaded.")
    return np.asarray(pcd.points), pcd

def compute_emd(depth_point_cloud, threed_point_cloud, i,output_folder):
    """
    Compute the Earth Mover's Distance (EMD) between two point clouds
    using the POT (Python Optimal Transport) library.
    """
    print(f"Computing EMD for image {i + 1}...")

    # Equalize the size of the point clouds
    min_points = min(len(depth_point_cloud), len(threed_point_cloud))
    depth_point_cloud = depth_point_cloud[:min_points]
    threed_point_cloud = threed_point_cloud[:min_points]

    plot_point_clouds(depth_point_cloud, threed_point_cloud, i,output_folder)

    # Euclidean distance calculation
    distance_matrix = cdist(depth_point_cloud, threed_point_cloud, metric='euclidean')

    # Weight normalized distribution
    weights1 = np.ones(len(depth_point_cloud)) / len(depth_point_cloud)
    weights2 = np.ones(len(threed_point_cloud)) / len(threed_point_cloud)

    # EMD calculation using optimal transport
    emd_value = ot.emd2(weights1, weights2, distance_matrix)

    print(f"EMD for image {i + 1} computed: {emd_value:.9f}")
    return emd_value

def plot_point_clouds(pc1, pc2, i,output_folder):
    print(f"Plotting point clouds for image {i + 1}...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], label='Depth cloud points', alpha=0.5)
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], label='3D model cloud points', alpha=0.5)
    ax.set_title("Cloud points comparison")
    plt.legend()
    plt.savefig(f"{output_folder}/combined_{i}.png")
    plt.close()
    print(f"Point clouds for image {i + 1} plotted and saved.")

def visualize_combined_clouds(model_cloud, depth_cloud, i):
    """
    Visualize the 3D model and depth map point clouds together.
    """
    print(f"Visualizing combined point clouds for image {i + 1}...")

    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(model_cloud)
    model_pcd.paint_uniform_color([0, 0, 1])  # Blue for the model

    depth_pcd = o3d.geometry.PointCloud()
    depth_pcd.points = o3d.utility.Vector3dVector(depth_cloud)
    depth_pcd.paint_uniform_color([1, 0, 0])  # Red for the depth map

    print(f"Visualization for image {i + 1} complete.")

def process_entity(entity_folder, model_3d_path, output_folder, camera_params):
    """
    Process a single entity to generate depth maps, compute EMD, and visualize results.
    """
    print(f"Processing entity in folder: {entity_folder}...")

    os.makedirs(output_folder, exist_ok=True)

    depth_estimator = DepthEstimator()

    # Load the 3D model and normalize it
    model_3d_cloud, _ = load_3d_model(model_3d_path)
    model_3d_cloud = normalize_point_cloud(model_3d_cloud)
    model_3d_cloud = sample_points(model_3d_cloud, 5000)  # Ensure it has 5000 points

    image_paths = [os.path.join(entity_folder, f) for f in os.listdir(entity_folder) if f.endswith(('.png', '.jpg', '.jpeg','.PNG','.JPG'))]

    emd_values = []

    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i + 1}/{len(image_paths)}: {image_path}...")
        image = cv2.imread(image_path)

        # Generate depth map
        depth_map = depth_estimator.predict_depth(image)

        # Convert depth map to point cloud
        depth_cloud = depth_map_to_point_cloud(
            depth_map,
            fl_x=camera_params["fl_x"],
            fl_y=camera_params["fl_y"],
            cx=camera_params["cx"],
            cy=camera_params["cy"]
        )
        depth_cloud = normalize_point_cloud(depth_cloud)
        depth_cloud = sample_points(depth_cloud, 10000)  # Match the model 3D points

        # Compute EMD
        emd_value = compute_emd(depth_cloud, model_3d_cloud, i,output_folder)
        emd_values.append((os.path.basename(image_path), emd_value))

        # Save depth map visualization
        plt.imshow(depth_map, cmap="plasma")
        plt.colorbar()
        plt.title(f"Depth Map: {os.path.basename(image_path)}")
        plt.savefig(os.path.join(output_folder, f"depth_map_{i + 1}.png"))
        plt.close()

        # Visualize combined point clouds
        visualize_combined_clouds(model_3d_cloud, depth_cloud, i)

    # Save EMD values to CSV
    csv_path = os.path.join(output_folder, "emd_values.csv")
    print(f"Saving EMD values to {csv_path}...")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "EMD Value"])
        writer.writerows(emd_values)

    # Compute and print average EMD
    average_emd = np.mean([value[1] for value in emd_values])
    print(f"Average EMD: {average_emd:.9f}")
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Average EMD", average_emd])

if __name__ == "__main__":
    camera_params = {
            "fl_x": 500,
            "fl_y": 500,
            "cx": 320,
            "cy": 240,
            "w": 4032.0,
            "h": 4032.0
        }

    entity_folders = [os.path.join(root_folder, entity) for entity in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, entity))]

    for entity_folder in entity_folders:
        
        entity_output_folder = f"{entity_folder}/output/"
        model_3d_path = f"{entity_folder}/{entity_folder.split('/')[6]}.obj"
        print('entity_folder',entity_folder)
        print('model_3d_path',model_3d_path)
        print('entity_output_folder',entity_output_folder)
        entity_folder = f"{entity_folder}/images"
        process_entity(entity_folder, model_3d_path, entity_output_folder, camera_params)


