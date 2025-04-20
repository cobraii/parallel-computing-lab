#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <queue>
#include <atomic>

#define MAX_FRAMES 100 
#define MAX_THREADS 4  

std::mutex mtx; 

void process_frame(const cv::Mat& frame, cv::CascadeClassifier& face_cascade, 
                  cv::CascadeClassifier& eye_cascade, cv::CascadeClassifier& smile_cascade,
                  std::vector<cv::Rect>& faces, std::vector<cv::Rect>& eyes, 
                  std::vector<cv::Rect>& smiles, int frame_id) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

    for (const auto& face : faces) {
        cv::Mat faceROI = gray(face);
        std::vector<cv::Rect> local_eyes, local_smiles;
        eye_cascade.detectMultiScale(faceROI, local_eyes, 1.1, 2, 0, cv::Size(20, 20));
        smile_cascade.detectMultiScale(faceROI, local_smiles, 1.3, 5, 0, cv::Size(30, 30));

        for (auto& eye : local_eyes) {
            eye.x += face.x;
            eye.y += face.y;
            eyes.push_back(eye);
        }
        for (auto& smile : local_smiles) {
            smile.x += face.x;
            smile.y += face.y;
            smiles.push_back(smile);
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Frame " << frame_id << ": " << faces.size() << " faces, "
                  << eyes.size() << " eyes, " << smiles.size() << " smiles\n";
    }
}

double process_sequential(cv::VideoCapture& cap, cv::CascadeClassifier& face_cascade,
                         cv::CascadeClassifier& eye_cascade, cv::CascadeClassifier& smile_cascade) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat frame;
    int frame_count = 0;

    while (frame_count < MAX_FRAMES && cap.read(frame)) {
        std::vector<cv::Rect> faces, eyes, smiles;
        process_frame(frame, face_cascade, eye_cascade, smile_cascade, faces, eyes, smiles, frame_count);

        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }
        for (const auto& eye : eyes) {
            cv::rectangle(frame, eye, cv::Scalar(255, 0, 0), 1);
        }
        for (const auto& smile : smiles) {
            cv::rectangle(frame, smile, cv::Scalar(0, 0, 255), 1);
        }
        cv::imshow("Sequential", frame);
        cv::waitKey(1);
        frame_count++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double process_parallel(cv::VideoCapture& cap, cv::CascadeClassifier& face_cascade,
                       cv::CascadeClassifier& eye_cascade, cv::CascadeClassifier& smile_cascade) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    std::queue<cv::Mat> frame_queue;
    std::vector<std::vector<cv::Rect>> all_faces(MAX_FRAMES), all_eyes(MAX_FRAMES), all_smiles(MAX_FRAMES);
    std::atomic<int> frame_count(0);
    int processed_frames = 0;

    auto worker = [&]() {
        while (true) {
            cv::Mat frame;
            int id;
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (frame_queue.empty() || frame_count >= MAX_FRAMES) return;
                frame = frame_queue.front();
                frame_queue.pop();
                id = frame_count++;
            }
            std::vector<cv::Rect> faces, eyes, smiles;
            process_frame(frame, face_cascade, eye_cascade, smile_cascade, faces, eyes, smiles, id);
            all_faces[id] = faces;
            all_eyes[id] = eyes;
            all_smiles[id] = smiles;
        }
    };

    cv::Mat frame;
    while (processed_frames < MAX_FRAMES && cap.read(frame)) {
        frame_queue.push(frame.clone());
        processed_frames++;
    }

    for (int i = 0; i < MAX_THREADS; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Сброс видео
    for (int i = 0; i < processed_frames; ++i) {
        if (!cap.read(frame)) break;
        for (const auto& face : all_faces[i]) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }
        for (const auto& eye : all_eyes[i]) {
            cv::rectangle(frame, eye, cv::Scalar(255, 0, 0), 1);
        }
        for (const auto& smile : all_smiles[i]) {
            cv::rectangle(frame, smile, cv::Scalar(0, 0, 255), 1);
        }
        cv::imshow("Parallel", frame);
        cv::waitKey(1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    cv::CascadeClassifier face_cascade, eye_cascade, smile_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml") ||
        !eye_cascade.load("haarcascade_eye.xml") ||
        !smile_cascade.load("haarcascade_smile.xml")) {
        std::cerr << "Error loading cascade files\n";
        return 1;
    }

    cv::VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        std::cout << "Failed to open video file, trying webcam\n";
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video capture\n";
            return 1;
        }
    }

    std::cout << "Running sequential processing...\n";
    double seq_time = process_sequential(cap, face_cascade, eye_cascade, smile_cascade);
    std::cout << "Sequential time: " << seq_time << " seconds\n";

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    std::cout << "Running parallel processing...\n";
    double par_time = process_parallel(cap, face_cascade, eye_cascade, smile_cascade);
    std::cout << "Parallel time: " << par_time << " seconds\n";

    double speedup = seq_time / par_time;
    std::cout << "Speedup: " << speedup << "\n";

    cv::destroyAllWindows();
    return 0;
}