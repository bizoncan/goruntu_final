import cv2
import numpy as np
import xml.etree.ElementTree as ET

def register_images(ref_image, test_image, ratio_thresh=0.7):
    # SIFT algılama
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(ref_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(test_image, None)

    if descriptors1 is None or descriptors2 is None:
        print("SIFT failed to find descriptors.")
        return None

    # FLANN tabanlı eşleştirici
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # İyi eşleşmeleri seç
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) == 0:
        print("No good matches found.")
        return None

    # Keypoint'lerden koordinatları çıkar
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Homografi matrisini hesapla
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography calculation failed.")
        return None

    height, width = ref_image.shape
    registered_image = cv2.warpPerspective(test_image, H, (width, height))

    return registered_image

def detect_defects(ref_image, test_image, diff_thresh=10, morph_kernel_size=1, area_threshold=100):
    # Fark görüntüsü oluştur
    diff_image = cv2.absdiff(ref_image, test_image)
    
    # İkili eşikleme
    _, thresh_image = cv2.threshold(diff_image, diff_thresh, 255, cv2.THRESH_BINARY)
    
    # Gürültü temizleme
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)
    
    # Konturları bul
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Kusurları işaretle
    defect_image = ref_image.copy()
    cv2.drawContours(defect_image, contours, -1, (0, 0, 255), 2)

    detected_defects = []
    # Kontur sayısını ve min-max koordinatlarını yazdır
    print("Detected defects:")
    defect_counter = 1
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            x_coords = [point[0][0] for point in contour]
            y_coords = [point[0][1] for point in contour]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            print(f"Defect {defect_counter}:")
            print(f"   Min Coordinate: ({min_x}, {min_y})")
            print(f"   Max Coordinate: ({max_x}, {max_y})")
            detected_defects.append((min_x, min_y, max_x, max_y))
            defect_counter += 1

    if defect_counter == 1:
        print("No defects detected.")
    return defect_image, detected_defects

def parse_defects_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_defects = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        min_x = int(bndbox.find('xmin').text)
        min_y = int(bndbox.find('ymin').text)
        max_x = int(bndbox.find('xmax').text)
        max_y = int(bndbox.find('ymax').text)
        xml_defects.append((min_x, min_y, max_x, max_y))
    return xml_defects

def compare_defects(detected_defects, xml_defects, threshold=10):
    correct_detections = 0
    for detected in detected_defects:
        for xml_defect in xml_defects:
            if (abs(detected[0] - xml_defect[0]) <= threshold and
                abs(detected[1] - xml_defect[1]) <= threshold and
                abs(detected[2] - xml_defect[2]) <= threshold and
                abs(detected[3] - xml_defect[3]) <= threshold):
                correct_detections += 1
                break
    return correct_detections

# Ana fonksiyon
def main(ref_image_path, test_image_path, xml_path, diff_thresh=10, morph_kernel_size=1, area_threshold=100, ratio_thresh=0.7):
    # Görüntüleri yükle
    ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    if ref_image is None:
        print(f"Failed to load reference image from {ref_image_path}")
        return
    if test_image is None:
        print(f"Failed to load test image from {test_image_path}")
        return
    
    # Görüntüleri hizala
    registered_test_image = register_images(ref_image, test_image, ratio_thresh)
    if registered_test_image is None:
        print("Image registration failed.")
        return
    
    # Kusurları tespit et
    defect_image, detected_defects = detect_defects(ref_image, registered_test_image, diff_thresh, morph_kernel_size, area_threshold)
    
    # XML'den kusurları yükle
    xml_defects = parse_defects_from_xml(xml_path)
    
    # Kusurları karşılaştır
    correct_detections = compare_defects(detected_defects, xml_defects, threshold=30)
    print(f"Correctly detected defects: {correct_detections} / {len(xml_defects)}")
    
    # Görüntüleri 1280x720 çözünürlüğe yeniden boyutlandır
    registered_test_image_resized = cv2.resize(registered_test_image, (1280, 720))
    defect_image_resized = cv2.resize(defect_image, (1280, 720))
    
    # Sonuçları göster
    cv2.imshow("Registered Test Image", registered_test_image_resized)
    cv2.imshow("Defects", defect_image_resized)
    
    # Klavyeden herhangi bir tuşa basıldığında işlemi sonlandır
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Dosya yollarını belirleyin
ref_image_path = r"C:\Users\New\Desktop\Yazilim\goruntu_final\01.JPG"
test_image_path = r'C:\Users\New\Desktop\Yazilim\goruntu_final\01_missing_hole_01.jpg'
xml_path = r'C:\Users\New\Desktop\Yazilim\goruntu_final\01_missing_hole_01.xml'

# Ana fonksiyonu çalıştır
main(ref_image_path, test_image_path, xml_path, diff_thresh=10, morph_kernel_size=1, area_threshold=50)