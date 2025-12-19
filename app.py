import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os
import random
from rembg import remove
import io
import numpy as np

# AYARLAR 
MODEL_PATH = 'model.pth' # .pth dosyanın adı
NUM_CLASSES = 23         # 22 Bitki + 1 Others

DISEASE_INFO = {
    # DOMATES (TOMATO)
    'Tomato___Bacterial_spot': {
        'name': 'Domates: Bakteriyel Leke (Bacterial Spot)',
        'cause': 'Xanthomonas bakterisi neden olur. Özellikle yüksek nemli ve yağışlı havalarda hızla yayılır.',
        'prevention': 'Sertifikalı ve temiz tohum kullanın. Damlama sulama tercih edin (yaprakları ıslatmayın).',
        'treatment': 'Hastalık görülür görülmez Bakır içerikli preparatlar uygulanmalıdır. Hastalıklı bitkiler sökülüp imha edilmelidir.'
    },
    'Tomato___Early_blight': {
        'name': 'Domates: Erken Yanıklık (Early Blight)',
        'cause': 'Alternaria solani mantarı. Genelde yaşlı yapraklarda "hedef tahtası" şeklinde halkalı lekeler yapar.',
        'prevention': 'Her yıl aynı yere domates ekmeyin (Ekim nöbeti/Münavebe). Bitkiler arası hava sirkülasyonunu artırın.',
        'treatment': 'Mancozeb, Chlorothalonil veya Azoxystrobin içeren fungisitler kullanılabilir.'
    },
    'Tomato___Late_blight': {
        'name': 'Domates: Geç Yanıklık (Late Blight)',
        'cause': 'Phytophthora infestans. Serin ve nemli havaları sever. Çok agresiftir, bitkiyi kısa sürede öldürebilir.',
        'prevention': 'Seraları sık sık havalandırın, nemi düşürün. Yaprakların uzun süre ıslak kalmasını önleyin.',
        'treatment': 'Hastalık belirtisi görülmeden koruyucu ilaçlama şarttır. Metalaxyl veya Mancozeb etkili olabilir.'
    },
    'Tomato___Leaf_Mold': {
        'name': 'Domates: Yaprak Küfü',
        'cause': 'Passalora fulva mantarı. Özellikle havalandırması kötü seralarda, yüksek nemde (%85+) ortaya çıkar.',
        'prevention': 'Sık ekimden kaçının, alt yaprakları budayarak havalandırmayı sağlayın.',
        'treatment': 'Kükürtlü ilaçlar veya uygun fungisitler ile ilaçlama yapılmalıdır.'
    },
    'Tomato___Septoria_leaf_spot': {
        'name': 'Domates: Septoria Yaprak Lekesi',
        'cause': 'Septoria lycopersici mantarı. Yapraklarda ortası gri, kenarı siyah küçük lekeler oluşturur.',
        'prevention': 'Yabancı otları temizleyin, bulaşık bitki artıklarını tarladan uzaklaştırıp yakın.',
        'treatment': 'Bakır bazlı fungisitler veya Klorotalonil içerikli ilaçlar uygulanabilir.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'name': 'Domates: İki Noktalı Kırmızı Örümcek',
        'cause': 'Sıcak ve kuru hava koşullarında üreyen mikroskobik zararlılardır (Akar). Yaprak özsuyunu emerler.',
        'prevention': 'Tarlayı ve bitki çevresini nemli tutmaya çalışın, tozlanmayı önleyin.',
        'treatment': 'Spesifik akarisitler (örümcek ilacı) veya Kükürt uygulaması yapılmalıdır.'
    },
    'Tomato___Target_Spot': {
        'name': 'Domates: Hedef Leke Hastalığı',
        'cause': 'Corynespora cassiicola mantarı. Yapraklarda iç içe geçmiş halkalar şeklinde lekeler yapar.',
        'prevention': 'Aşırı azotlu gübrelemeden kaçının. Hava akımını sağlamak için budama yapın.',
        'treatment': 'Azoxystrobin veya Boscalid içeren sistemik fungisitler kullanılabilir.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Domates: Sarı Yaprak Kıvırcıklığı Virüsü',
        'cause': 'Virüstür ve sadece "Beyaz Sinek" (Bemisia tabaci) tarafından taşınır. Yapraklar sararır ve kıvrılır.',
        'prevention': 'Dayanıklı tohum kullanın. Seralara sinek tülleri takın ve Beyaz Sinek ile mücadele edin.',
        'treatment': 'Virüsün kimyasal tedavisi YOKTUR. Hasta bitkiyi kökünden söküp uzaklaştırın.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'name': 'Domates: Mozaik Virüsü',
        'cause': 'Mekanik yolla (insan eli, aletler, kıyafetler) veya enfekte tohumla bulaşır.',
        'prevention': 'Çalışırken elleri ve aletleri sık sık dezenfekte edin. Sigara içtikten sonra bitkiye dokunmayın (tütünden geçer).',
        'treatment': 'Tedavisi YOKTUR. Enfekte bitkileri hemen imha edin.'
    },
    'Tomato___healthy': {
        'name': 'Domates: Sağlıklı',
        'cause': '-',
        'prevention': '-',
        'treatment': '-'
    },

    # ELMA (APPLE)
    'Apple___Apple_scab': {
        'name': 'Elma: Kara Leke',
        'cause': 'Venturia inaequalis mantarı. İlkbaharın yağışlı ve serin gitmesi hastalığı tetikler.',
        'prevention': 'Sonbaharda dökülen yaprakları toplayıp yakın (mantar kışı orada geçirir).',
        'treatment': 'Tomurcuk kabarmasından itibaren düzenli bakırlı ve organik fungisit uygulaması gerekir.'
    },
    'Apple___Black_rot': {
        'name': 'Elma: Siyah Çürüklük',
        'cause': 'Botryosphaeria obtusa mantarı. Meyvede siyah çürümeler, yaprakta "kurbağa gözü" lekesi yapar.',
        'prevention': 'Ağaçtaki mumyalaşmış meyveleri toplayın. Yaralı ve kuru dalları budayın.',
        'treatment': 'Captan veya Thiophanate-methyl içeren ilaçlar kullanılabilir.'
    },
    'Apple___Cedar_apple_rust': {
        'name': 'Elma: Sedir Pası',
        'cause': 'Gymnosporangium mantarı. Hastalığın oluşması için yakında Ardıç (Sedir) ağacı olması gerekir.',
        'prevention': 'Bahçe yakınındaki ardıç ağaçlarını temizleyin veya dayanıklı elma çeşitleri seçin.',
        'treatment': 'İlkbaharda, çiçeklenme döneminde pas ilaçlaması (Myclobutanil vb.) yapılmalıdır.'
    },
    'Apple___healthy': {
        'name': 'Elma: Sağlıklı',
        'cause': '-',
        'prevention': '-',
        'treatment': '-'
    },

    # MISIR (CORN)
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'name': 'Mısır: Gri Yaprak Lekesi',
        'cause': 'Cercospora zeae-maydis mantarı. Dikdörtgen şeklinde gri/kahverengi lekeler yapar.',
        'prevention': 'Hastalığa dayanıklı hibrit tohumlar kullanın. Tarlada hasat artığı bırakmayın.',
        'treatment': 'Hastalık koçan püskülü döneminde görülürse fungisit uygulanabilir.'
    },
    'Corn_(maize)___Common_rust_': {
        'name': 'Mısır: Pas Hastalığı',
        'cause': 'Puccinia sorghi mantarı. Yaprağın iki yüzünde de kiremit kırmızısı kabarcıklar oluşur.',
        'prevention': 'Erken ekim yaparak bitkinin güçlenmesini sağlayın.',
        'treatment': 'Genelde ekonomik zarar eşiğini aşmazsa ilaçlama önerilmez, aşarsa fungisit atılır.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'name': 'Mısır: Kuzey Yaprak Yanıklığı',
        'cause': 'Exserohilum turcicum mantarı. Yapraklarda uzun, mekik (puro) şeklinde gri lekeler yapar.',
        'prevention': 'Ekim nöbeti (münavebe) uygulayın. Bir yıl mısır, bir yıl başka ürün ekin.',
        'treatment': 'Hastalık belirtileri erken dönemde görülürse ilaçlama yapılabilir.'
    },
    'Corn_(maize)___healthy': {
        'name': 'Mısır: Sağlıklı',
        'cause': '-',
        'prevention': '-',
        'treatment': '-'
    },

    # ÜZÜM (GRAPE)
    'Grape___Black_rot': {
        'name': 'Üzüm: Siyah Çürüklük',
        'cause': 'Guignardia bidwellii mantarı. Meyveleri büzüştürür ve mumyalaştırır (siyah kuru üzüm gibi olur).',
        'prevention': 'Kış budamasında hastalıklı dalları ve kurumuş salkımları bağdan uzaklaştırın.',
        'treatment': 'İlkbaharda sürgünler 10-15 cm olunca ilaçlamaya başlayın (Mancozeb, Bakır).'
    },
    'Grape___Esca_(Black_Measles)': {
        'name': 'Üzüm: Kav Hastalığı (Esca)',
        'cause': 'Çeşitli mantarların (Phaeomoniella vb.) neden olduğu bir gövde hastalığıdır. Yaprakta kaplan deseni yapar.',
        'prevention': 'Budama makaslarını dezenfekte edin. Büyük budama yaralarına aşı macunu sürün.',
        'treatment': 'Kesin bir kimyasal tedavisi yoktur. Hasta asmalar işaretlenip, gerekirse sökülmelidir.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'name': 'Üzüm: Yaprak Yanıklığı',
        'cause': 'Pseudocercospora vitis mantarı. Genelde hasat sonlarına doğru, zayıf düşmüş bağlarda görülür.',
        'prevention': 'Asmanın gübreleme ve su dengesine dikkat edin, bitkiyi güçlü tutun.',
        'treatment': 'Genelde hasada yakın olduğu için ilaçlama gerekmeyebilir, erken dönemde ise fungisit atılır.'
    },
    'Grape___healthy': {
        'name': 'Üzüm: Sağlıklı',
        'cause': '-',
        'prevention': '-',
        'treatment': '-'
    },

    # OTHERS (DİĞERLERİ)
    'Others': {
        'name': 'Tanımlanamayan Nesne / Bitki Değil',
        'cause': 'Yüklenen fotoğraf, sistemin tanıdığı bitki türlerine (Domates, Mısır, Elma, Üzüm) ait değil.',
        'prevention': '-',
        'treatment': '-'
    }
}

# 2. SINIF İSİMLERİ
CLASS_NAMES = sorted(list(DISEASE_INFO.keys()))
if 'Others' in CLASS_NAMES:
    CLASS_NAMES.remove('Others')
    CLASS_NAMES.append('Others')

# SAYFA AYARLARI
st.set_page_config(page_title="Bitki Doktoru", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6 }
    .title { color: #2e7d32; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='title'>Akıllı Bitki Hastalık Tespit Sistemi</h1>", unsafe_allow_html=True)
st.write("Domates, Elma, Mısır ve Üzüm hastalıklarını yapay zeka ile teşhis edin.")

# MODEL YÜKLEME
@st.cache_resource
def load_model():
    try:
        device = torch.device('cpu')
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi! Hata: {e}")
        return None

model = load_model()

# GÖRÜNTÜ İŞLEME VE TEMİZLEME
def process_image(image, temizle=False):
    """
    Görüntüyü alır, isteğe bağlı olarak arka planı rembg ile siler
    ve model için tensor formatına çevirir.
    """
    # 1. ARKA PLAN TEMİZLEME
    if temizle:
        # PIL Image -> Bytes dönüşümü (Rembg için)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Arka planı kaldır
        output = remove(img_byte_arr)
        
        # Tekrar PIL Image'a çevir
        image = Image.open(io.BytesIO(output)).convert('RGB')
        
    # 2. MODEL İÇİN HAZIRLIK (Transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Hem tensoru hem de (varsa temizlenmiş) resmi döndür
    return transform(image).unsqueeze(0), image

# ARAYÜZ SEKMELERİ (TABS)
tab1, tab2 = st.tabs(["Hastalık Tahmini", "Nasıl Kullanılır?"])

with tab1:
    st.header("Fotoğraf Yükle")
    uploaded_file = st.file_uploader("Bir yaprak fotoğrafı seçin...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Resmi ortalayarak göster
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption='Analiz Edilecek Görüntü', use_container_width=True)
            
            # Checkbox
            arkaplan_temizle = st.checkbox("Arka Planı Temizle (Daha net sonuç için)", value=True)
            
            predict_btn = st.button('Hastalığı Teşhis Et', use_container_width=True)

        if predict_btn:
            with st.spinner('Arka plan siliniyor'):
                
                # Fonksiyonu yeni haliyle çağır
                img_tensor, islenmis_resim = process_image(image, temizle=arkaplan_temizle)
                
                # Eğer temizleme yapıldıysa temiz halini göster
                if arkaplan_temizle:
                     # Resmi biraz küçültüp göster
                     with col2:
                        st.image(islenmis_resim, caption="Arka Planı Temizlenmiş Görüntü", width=200)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = F.softmax(outputs[0], dim=0)
                
                top_prob, top_catid = torch.topk(probs, 1)
                guven = top_prob.item() * 100
                tahmin_index = top_catid.item()
                
                tahmin_sinif = CLASS_NAMES[tahmin_index]
                
                # SONUÇ GÖSTERİMİ
                if tahmin_sinif == 'Others':
                    st.warning(f"**Tanımlanamadı / Bitki Değil** (Güven: %{guven:.2f})")
                else:
                    info = DISEASE_INFO.get(tahmin_sinif, {})
                    
                    st.success(f"**Teşhis:** {info.get('name', tahmin_sinif)}")
                    st.progress(int(guven))
                    st.caption(f"Güven Oranı: %{guven:.2f}")

                    with st.expander("Hastalık Detayları ve Tedavi Yöntemleri", expanded=True):
                        st.markdown(f"""
                        **Neden Olur:** {info.get('cause', '-')}
                        
                        **Önleme:** {info.get('prevention', '-')}
                        
                        **Tedavi:** {info.get('treatment', '-')}
                        """)

with tab2:
    st.header("Sistem Hakkında")
    st.info("""
    Bu proje **ResNet50** mimarisi kullanılarak geliştirilmiştir.
    
    **Desteklenen Bitkiler:**
    * Domates (10 Sınıf)
    * Elma (4 Sınıf)
    * Mısır (4 Sınıf)
    * Üzüm (4 Sınıf)
    
    Model, **PlantVillage** veri seti ve **Natural Images** (Araba, İnsan vb. ayırt etme) verileriyle eğitilmiştir.
            
    M. Arif Dayı/Mehmet Emircan Küllücek
    """)
