import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class DinoWrapper:
    def __init__(self, device):
        self.device = device
        print(f"--> Ładowanie modelu DINOv2 (Small) na {self.device}...")
        
        # Ładowanie modelu z huba
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.to(self.device)
        self.model.eval()
        
        # Transformacje
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.prototypes = {}

    def _get_embedding(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Błąd otwierania {image_path}: {e}")
            return None

        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_t)
            
        # Normalizacja L2
        return F.normalize(features, p=2, dim=1)

    def fit_prototypes(self, data_dict):
        self.prototypes = {}
        print(f"--> Budowanie wzorców dla {len(data_dict)} klas...")

        for class_name, paths in data_dict.items():
            embeddings_list = []
            for p in paths:
                emb = self._get_embedding(p)
                if emb is not None:
                    embeddings_list.append(emb)
            
            if not embeddings_list:
                continue

            # Średnia wektorów (centroid)
            stack = torch.cat(embeddings_list, dim=0)
            mean_vector = torch.mean(stack, dim=0, keepdim=True)
            mean_vector = F.normalize(mean_vector, p=2, dim=1)
            
            self.prototypes[class_name] = mean_vector

    def predict(self, image_path):
        """
        Zwraca: (klasa, wynik, wektor_numpy)
        """
        if not self.prototypes:
            raise ValueError("Brak prototypów! Uruchom fit_prototypes.")

        query_emb = self._get_embedding(image_path)
        if query_emb is None:
            return "Error", 0.0, None

        best_score = -1.0
        best_class = None

        # Porównanie z wzorcami
        for cls_name, proto_emb in self.prototypes.items():
            similarity = torch.mm(query_emb, proto_emb.T).item()
            if similarity > best_score:
                best_score = similarity
                best_class = cls_name
                
        # Zwracamy wektor (na CPU) do wizualizacji t-SNE
        return best_class, best_score, query_emb.cpu().numpy()