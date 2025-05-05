# E-Waste Object Classification Model

This repository contains a YOLOv11-based object classification model for electronic waste (e-waste) items.

## Model Details

- **Framework**: YOLOv11
- **Classes**: 8 e-waste categories
- **Input**: RGB images
- **Output**: Bounding boxes and class predictions

### Classes
1. Laptop
2. Phone
3. Tablet
4. Keyboard
5. Mouse
6. Printer
7. Television
8. Washing Machine

## Directory Structure

```
object-classification-model/
├── data/               # Dataset (not included in repo)
│   ├── images/        # Training and validation images
│   └── labels/        # Annotation files
├── train/             # Training configuration
│   └── e-waste.yaml   # Dataset configuration
└── venv/              # Python virtual environment (not included)
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
- Place training images in `data/images/train`
- Place validation images in `data/images/val`
- Place corresponding label files in `data/labels`

4. Run Label Studio to annotate images:
```bash
label-studio
```
- Copy the Labelling Inference code settings :
```bash
<View>
  <Image name="image" value="$image" rotateControl="true" zoomControl="true" zoom="true"/>

  <RectangleLabels name="label" toName="image">
    <Label value="Kipas Rad /Box 10" background="#FFA39E"/>
    <Label value="Notebook Cooler" background="#D4380D"/>
    <Label value="Kipas X Cool" background="#FFC069"/>
    <Label value="Vga" background="#AD8B00"/>
    <Label value="Jam Dinding" background="#D3F261"/>
    <Label value="Seterika" background="#389E0D"/>
    <Label value="Kipas" background="#5CDBD3"/>
    <Label value="Lampu Gantung" background="#096DD9"/>
    <Label value="Blender" background="#ADC6FF"/>
    <Label value="Sinar Up I Hancur" background="#9254DE"/>
    <Label value="Kompresor Kulkas" background="#F759AB"/>
    <Label value="Antena" background="#FFA39E"/>
    <Label value="Vacum Cleaner" background="#D4380D"/>
    <Label value="Remot" background="#FFC069"/>
    <Label value="Microfon" background="#AD8B00"/>
    <Label value="Cooler" background="#D3F261"/>
    <Label value="Flashdisk" background="#389E0D"/>
    <Label value="Stik Ps" background="#5CDBD3"/>
    <Label value="Alat Tensi" background="#096DD9"/>
    <Label value="Remot Kontrol" background="#ADC6FF"/>
    <Label value="Mixer" background="#9254DE"/>
    <Label value="Kompor Listrik" background="#F759AB"/>
    <Label value="Router" background="#FFA39E"/>
    <Label value="Kamera Jadul" background="#D4380D"/>
    <Label value="Magicom Kecil" background="#FFC069"/>
    <Label value="Solder" background="#AD8B00"/>
    <Label value="Mouse" background="#D3F261"/>
    <Label value="Speaker Kecil" background="#389E0D"/>
    <Label value="Telefon" background="#5CDBD3"/>
    <Label value="Headset" background="#096DD9"/>
    <Label value="Seterika Plastik" background="#ADC6FF"/>
    <Label value="Alat Tes Vol" background="#9254DE"/>
    <Label value="Hardisk Laptop" background="#F759AB"/>
    <Label value="Modem" background="#FFA39E"/>
    <Label value="Wireless Charger" background="#D4380D"/>
    <Label value="CCTV" background="#FFC069"/>
    <Label value="Baterai Laptop" background="#AD8B00"/>
    <Label value="Jam Tangan Cina" background="#D3F261"/>
    <Label value="Monitor Semua Merk Rad" background="#389E0D"/>
    <Label value="Charger Laptop" background="#5CDBD3"/>
    <Label value="Panel Surya" background="#096DD9"/>
    <Label value="Career Pc" background="#ADC6FF"/>
    <Label value="Keyboard" background="#9254DE"/>
    <Label value="Kaset CD" background="#F759AB"/>
    <Label value="Teko Listrik" background="#FFA39E"/>
    <Label value="Blender Kecil" background="#D4380D"/>
    <Label value="Jam Digital" background="#FFC069"/>
    <Label value="Kipas Rad" background="#AD8B00"/>
    <Label value="VGA" background="#D3F261"/>
    <Label value="Ant Minter" background="#389E0D"/>
    <Label value="Hair Dryer" background="#5CDBD3"/>
    <Label value="Catokan" background="#096DD9"/>
    <Label value="Motherboard" background="#ADC6FF"/>
    <Label value="Power Bank" background="#9254DE"/>
    <Label value="Should Blaster" background="#F759AB"/>
    <Label value="Saklar Lampu 3 Rad" background="#FFA39E"/>
    <Label value="Kabel Sambungan" background="#D4380D"/>
    <Label value="DVD Room" background="#FFC069"/>
    <Label value="Speaker Bluetooth" background="#AD8B00"/>
    <Label value="Jam Tangan Rusak" background="#D3F261"/>
    <Label value="Box Kabel" background="#389E0D"/>
    <Label value="PsPs" background="#5CDBD3"/>
    <Label value="Timbangan Badan" background="#096DD9"/>
    <Label value="Lampu Surya" background="#ADC6FF"/>
    <Label value="Homic Wireless" background="#9254DE"/>
    <Label value="Radio" background="#F759AB"/>
    <Label value="Modem Xiaomi" background="#FFA39E"/>
    <Label value="Pompa Air Elektrik" background="#D4380D"/>
    <Label value="Mesin Kasir" background="#FFC069"/>
    <Label value="Raket Nyamuk" background="#AD8B00"/>
    <Label value="DVD Player" background="#D3F261"/>
    <Label value="Mesin Facial" background="#389E0D"/>
    <Label value="Mesin Fax" background="#5CDBD3"/>
    <Label value="Stop Kontak" background="#096DD9"/>
    <Label value="Mesin Pijat" background="#ADC6FF"/>
    <Label value="Aki Motor" background="#9254DE"/>
    <Label value="Lampu Belajar" background="#F759AB"/>
    <Label value="Lampu Tidur" background="#FFA39E"/>
    <Label value="Kabel 1 Box/Kilo" background="#D4380D"/>
    <Label value="Senter" background="#FFC069"/>
    <Label value="Kabel /Kilo" background="#AD8B00"/>
    <Label value="Lampu LCD" background="#D3F261"/>
    <Label value="Sound Blaster" background="#389E0D"/>
    <Label value="Motherboard /Kilo" background="#5CDBD3"/>
    <Label value="Ant Miner Hash Board" background="#096DD9"/>
    <Label value="Digital Camera Samsung" background="#ADC6FF"/>
    <Label value="Power Supply Rad Semua Merk" background="#9254DE"/>
    <Label value="Mesin Sanyo" background="#F759AB"/>
    <Label value="Box Speaker Kecil" background="#FFA39E"/>
    <Label value="HP Tablet" background="#D4380D"/>
    <Label value="Nokia A" background="#FFC069"/>
    <Label value="Bb B" background="#AD8B00"/>
    <Label value="Bb A" background="#D3F261"/>
    <Label value="Android A" background="#389E0D"/>
    <Label value="Mesin Blender" background="#5CDBD3"/>
    <Label value="Hardisk" background="#096DD9"/>
    <Label value="Abal2 /Kilo" background="#ADC6FF"/>
    <Label value="Walkie Talkie" background="#9254DE"/>
    <Label value="Tinta" background="#F759AB"/>
    <Label value="Multi Tester" background="#FFA39E"/>
    <Label value="Laptop Jadul Kecil" background="#D4380D"/>
    <Label value="Printer" background="#FFC069"/>
    <Label value="Adaptor /Kilo" background="#AD8B00"/>
    <Label value="Power Supply" background="#D3F261"/>
    <Label value="Casing CPU" background="#389E0D"/>
    <Label value="CPU kecil" background="#5CDBD3"/>
    <Label value="Mesin Jahit" background="#096DD9"/>
    <Label value="Monitor Tabung" background="#ADC6FF"/>
    <Label value="TV Tabung" background="#9254DE"/>
    <Label value="DVD" background="#F759AB"/>
    <Label value="Bantal Pemanas" background="#FFA39E"/>
    <Label value="Sega" background="#D4380D"/>
    <Label value="Speaker Aktif Sedang" background="#FFC069"/>
    <Label value="Tabung Debu" background="#AD8B00"/>
    <Label value="AC" background="#D3F261"/>
    <Label value="Lampu Emergency" background="#389E0D"/>
    <Label value="Printer Portable" background="#5CDBD3"/>
    <Label value="Stabilizer" background="#096DD9"/>
    <Label value="Ultrasonic" background="#ADC6FF"/>
    <Label value="Oven Besar" background="#9254DE"/>
    <Label value="Microwave" background="#F759AB"/>
    <Label value="TV LED Rad" background="#FFA39E"/>
    <Label value="Mesin Cuci Tabung Tengah" background="#D4380D"/>
    <Label value="Mesin Cuci Tabung Atas Rad" background="#FFC069"/>
    <Label value="Kulkas" background="#AD8B00"/>
    <Label value="Ups" background="#D3F261"/>
    <Label value="AC Kecil" background="#389E0D"/>
    <Label value="Batok Charger" background="#5CDBD3"/>
    <Label value="Kabel Headset" background="#096DD9"/>
    <Label value="Neon Box" background="#ADC6FF"/>
    <Label value="Mesin Cuci Motor" background="#9254DE"/>
    <Label value="Payung Lampu" background="#F759AB"/>
    <Label value="Dudukan Kulkas" background="#FFA39E"/>
  </RectangleLabels>

  <TextArea name="price" toName="image" editable="true" required="true" perRegion="true" placeholder="Enter price in IDR"/>

  <Header value="Ukuran (Size)" />
  <Choices name="size" toName="image" perRegion="true">
    <Choice value="small"/>
    <Choice value="medium"/>
    <Choice value="large"/>
  </Choices>

  <Header value="Kondisi Debu dan Karat (Dust and Rust)" />
  <Choices name="dust-rust" toName="image" perRegion="true">
    <Choice value="clean"/>
    <Choice value="dirty"/>
    <Choice value="very-dirty"/>
  </Choices>
  
  <Header value="Kondisi Barang (Condition)" />
  <Choices name="condition" toName="image" perRegion="true">
    <Choice value="good"/>
    <Choice value="scratched"/>
    <Choice value="damaged"/>
  </Choices>
</View>
```
- Paste in Labelling Inference code settings in Label Studio


## Training

To train the model:
```bash
yolo detect train model=yolov11n.pt data=train/e-waste.yaml epochs=100 imgsz=640
```

## Inference

To run inference on new images:
```bash
yolo predict model=path/to/best.pt source=path/to/image.jpg
```

## Notes

- Download Dataset [here](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset)
- Model weights and checkpoints should be stored separately
- Use the provided YAML configuration for training
