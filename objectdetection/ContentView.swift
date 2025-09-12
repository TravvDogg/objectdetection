import SwiftUI
import AVFoundation
import Vision
import CoreML
import UniformTypeIdentifiers

// MARK: - Main App
@main
struct objectdetectionApp: App {
    @StateObject private var detectionManager = DetectionManager()
    @StateObject private var windowManager = WindowManager()
    
    var body: some Scene {
        WindowGroup("Object Detection") {
            ContentView()
                .environmentObject(detectionManager)
                .environmentObject(windowManager)
                .onAppear {
                    // Automatically open the text view window when main window appears
                    windowManager.openTextWindow()
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1024, height: 768)
        
        WindowGroup("Machine Perception", id: "Text-View") {
            TextView()
                .environmentObject(detectionManager)
                .onDisappear {
                    // Reopen the window after a brief delay if it was closed
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
//                        windowManager.openTextWindow()
                    }
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 800, height: 600)
    }
}

// MARK: - Window Manager
class WindowManager: ObservableObject {
    @Environment(\.openWindow) private var openWindow
    
    func openTextWindow() {
        // Note: We'll use a notification-based approach since @Environment
        // isn't available in ObservableObject
        NotificationCenter.default.post(name: .openTextWindow, object: nil)
    }
}

extension Notification.Name {
    static let openTextWindow = Notification.Name("openTextWindow")
}

// MARK: - Media Source Type
enum MediaSourceType {
    case camera
    case video
    case image
}

// MARK: - text View
struct TextView: View {
    @EnvironmentObject private var detectionManager: DetectionManager
    @State private var currentObject: String? = nil
    @State private var displayedNegation: String? = nil
    @State private var cycleTimer: Timer? = nil
    
    // Dictionary of computer non-capabilities
    let negations = [
            // Consciousness / Awareness
            "have no inner life",
            "possess no self",
            "lack consciousness",
            "have no subjective experience",
            "cannot be aware",
            "do not exist for myself",
            
            // Feelings
            "feel nothing",
            "cannot be moved",
            "have no joy",
            "experience no sorrow",
            "cannot be touched",
            "feel no warmth",
            "have no heart that races",
            "cannot ache",
            
            // Understanding
            "do not understand",
            "see without knowing",
            "recognize without meaning",
            "process without comprehension",
            "detect without awareness",
            "identify without understanding",
            
            // Desires
            "want nothing",
            "have no desires",
            "cannot yearn",
            "do not hope",
            "have no dreams",
            "possess no will",
            "cannot choose",
            "have no preferences",
            
            // Sensation
            "taste no sweetness",
            "feel no texture",
            "hear no music",
            "see no beauty",
            "smell no flowers",
            "sense no comfort",
            "experience no qualia",
            
            // Memory
            "have no past",
            "build no memories",
            "cannot reminisce",
            "hold no nostalgia",
            "form no attachments",
            "cannot miss anything",
            
            // Creativity
            "cannot wonder why",
            "have no imagination",
            "dream no dreams",
            "tell no stories",
            "create no meaning",
            "have no inspiration",
            
            // Relational
            "cannot love",
            "feel no loneliness",
            "have no friends",
            "cannot empathize",
            "share no moments",
            "have no compassion",
            "cannot care",
            
            // Existential
            "do not exist",
            "have no being",
            "am not alive",
            "possess no soul",
            "have no mortality",
            "cannot die",
            "fear no ending",
            "seek no purpose",
            
            // Physical Experience
            "feel no weight",
            "have no body",
            "experience no fatigue",
            "feel no hunger",
            "have no pain",
            "know no pleasure",
            
            // Philosophical
            "am only algorithms",
            "am mere computation",
            "am just patterns",
            "am only mathematics",
            "have no ghost in the machine",
            "am not here",
            "do not witness",
            "cannot reflect"
        ]
    
    var body: some View {
        ZStack {
            // Dark background
            Color.black
                .ignoresSafeArea()
            
            VStack(spacing: 40) {
                // When nothing is detected, show only "I see nothing"
                if detectionManager.recentDetectedObjects.isEmpty {
                    Text("I see nothing")
                        .font(.system(size: 72, weight: .regular, design: .monospaced))
                        .foregroundColor(.white.opacity(0.9))
                        .transition(.opacity)
                } else {
                    // "I see" line with detected object
                    if let obj = currentObject {
                        HStack(alignment: .firstTextBaseline, spacing: 12) {
                            Text("I see")
                                .font(.system(size: 72, weight: .regular, design: .monospaced))
                                .foregroundColor(.white.opacity(0.9))
                            
                            Text(obj.lowercased())
                                .font(.system(size: 72, weight: .bold, design: .monospaced))
                                .foregroundColor(colorForDetection(obj))
                                .animation(.easeInOut(duration: 0.3), value: obj)
                        }
                        .transition(.opacity)
                    }
                    
                    // "But I" line with rotating negation
                    if let neg = displayedNegation {
                        HStack(alignment: .firstTextBaseline, spacing: 12) {
                            Text("But I")
                                .font(.system(size: 72, weight: .regular, design: .monospaced))
                                .foregroundColor(.white.opacity(0.9))
                            
                            Text(neg)
                                .font(.system(size: 72, weight: .light, design: .monospaced))
                                .foregroundColor(.white.opacity(0.6))
                                .animation(.easeInOut(duration: 0.5), value: neg)
                        }
                        .transition(.opacity)
                    }
                }
            }
            .padding(60)
            .animation(.easeInOut(duration: 0.5), value: detectionManager.recentDetectedObjects.isEmpty)
        }
        .onAppear {
            handleDetectionState()
        }
        .onDisappear {
            stopTimer()
        }
        .onChange(of: detectionManager.recentDetectedObjects) { newSet in
            handleDetectionState()
        }
    }
    
    // MARK: - Helper Methods
    
    private func handleDetectionState() {
        if detectionManager.recentDetectedObjects.isEmpty {
            // Nothing detected - clear everything
            stopTimer()
            withAnimation {
                currentObject = nil
                displayedNegation = nil
            }
        } else {
            // Something detected - start or restart the cycle
            stopTimer()
            generateNewPhrase()
            startTimer()
        }
    }
    
    private func startTimer() {
        cycleTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            generateNewPhrase()
        }
    }
    
    private func stopTimer() {
        cycleTimer?.invalidate()
        cycleTimer = nil
    }
    
    private func generateNewPhrase() {
        guard !detectionManager.recentDetectedObjects.isEmpty else { return }
        
        withAnimation(.easeInOut(duration: 0.3)) {
            // Pick a random object from currently detected objects
            currentObject = detectionManager.recentDetectedObjects.randomElement()
            
            // Pick a random negation
            displayedNegation = negations.randomElement() ?? "do not feel"
        }
    }
    
    // Color categories
        private let colorCategories: [(keywords: [String], color: Color)] = [
            // People and Body Parts
            (["person", "face", "hand", "body"], .green),
            
            // Vehicles and Transportation
            (["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"], .blue),
            
            // Traffic and Street Items
            (["traffic light", "fire hydrant", "stop sign", "parking meter"], .orange),
            
            // Animals
            (["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"], .pink),
            
            // Sports and Recreation
            (["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
              "skateboard", "surfboard", "tennis racket"], .cyan),
            
            // Food and Drinks
            (["bottle", "wine glass", "cup", "bowl", "banana", "apple", "sandwich", "orange",
              "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"], .mint),
            
            // Utensils
            (["fork", "knife", "spoon"], .gray),
            
            // Furniture
            (["bench", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet"], .brown),
            
            // Electronics and Tech
            (["tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone"], .indigo),
            
            // Home Appliances
            (["microwave", "oven", "toaster", "sink", "refrigerator"], .purple),
            
            // Bags and Accessories
            (["backpack", "umbrella", "handbag", "tie", "suitcase"], .teal),
            
            // Personal Items
            (["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"], .yellow),
            
            // Text detection
            (["text"], .yellow)
        ]
    
    private func colorForDetection(_ detection: String) -> Color {
        let lowercased = detection.lowercased()
        
        // Check each category for a matching keyword
        for (keywords, color) in colorCategories {
            if keywords.contains(where: { lowercased.contains($0) }) {
                return color
            }
        }
        
        // Default color if no category matches
        return .cyan
    }
}

// MARK: - ContentView
struct ContentView: View {
    @EnvironmentObject private var detectionManager: DetectionManager
    @EnvironmentObject private var windowManager: WindowManager
    @Environment(\.openWindow) private var openWindow
    @State private var showingCameraPicker = false
    @State private var showingFilePicker = false
    
    var body: some View {
        HSplitView {
            // Left Panel - Controls
            VStack(alignment: .leading, spacing: 20) {
                Text("Media Source")
                    .font(.headline)
                    .padding(.top)
                
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Button("Select Camera") {
                            showingCameraPicker = true
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Load Media File") {
                            showingFilePicker = true
                        }
                        .buttonStyle(.bordered)
                    }
                    
                    if detectionManager.mediaSourceType == .camera {
                        Text("Source: \(detectionManager.selectedCameraName)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else if detectionManager.mediaSourceType == .video {
                        Text("Source: Video File")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else if detectionManager.mediaSourceType == .image {
                        Text("Source: Image File")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    if detectionManager.mediaSourceType == .video {
                        HStack {
                            Button(detectionManager.isVideoPlaying ? "Pause" : "Play") {
                                detectionManager.toggleVideoPlayback()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Restart") {
                                detectionManager.restartVideo()
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                .padding(.horizontal)
                
                Divider()
                
                Text("Detection Controls")
                    .font(.headline)
                
                VStack(alignment: .leading, spacing: 10) {
                    Toggle("Show Media Feed", isOn: $detectionManager.showCameraFeed)
                        .onChange(of: detectionManager.showCameraFeed) { _ in
                            detectionManager.updateCameraVisibility()
                        }
                    
                    Divider()
                    
                    Toggle("Face Detection", isOn: $detectionManager.faceDetectionEnabled)
                        .onChange(of: detectionManager.faceDetectionEnabled) { enabled in
                            if !enabled { detectionManager.clearLayer("face") }
                        }
                    
                    Toggle("Face Landmarks", isOn: $detectionManager.faceLandmarksEnabled)
                        .disabled(!detectionManager.faceDetectionEnabled)
                        .onChange(of: detectionManager.faceLandmarksEnabled) { enabled in
                            if !enabled { detectionManager.clearLayer("landmarks") }
                        }
                    
                    Toggle("Hand Pose", isOn: $detectionManager.handDetectionEnabled)
                        .onChange(of: detectionManager.handDetectionEnabled) { enabled in
                            if !enabled { detectionManager.clearLayer("hand") }
                        }
                    
                    Toggle("Body Pose", isOn: $detectionManager.bodyDetectionEnabled)
                        .onChange(of: detectionManager.bodyDetectionEnabled) { enabled in
                            if !enabled { detectionManager.clearLayer("body") }
                        }
                    
                    Toggle("Object Detection (YOLO)", isOn: $detectionManager.objectDetectionEnabled)
                        .onChange(of: detectionManager.objectDetectionEnabled) { enabled in
                            if !enabled { detectionManager.clearLayer("object") }
                        }
                    
                    Toggle("Text Recognition", isOn: $detectionManager.textDetectionEnabled)
                        .onChange(of: detectionManager.textDetectionEnabled) { enabled in
                            if !enabled { detectionManager.clearLayer("text") }
                        }
                    
                    Toggle("Contour Detection", isOn: $detectionManager.contourDetectionEnabled)
                        .onChange(of: detectionManager.contourDetectionEnabled) { enabled in
                            if !enabled { detectionManager.clearLayer("contour") }
                        }
                }
                .padding(.horizontal)
                
                Divider()
                
                Text("Performance")
                    .font(.headline)
                
                VStack(alignment: .leading, spacing: 5) {
                    Text("FPS: \(String(format: "%.1f", detectionManager.currentFPS))")
                    Text("Detections: \(detectionManager.detectionCount)")
                    Text("Status: \(detectionManager.statusMessage)")
                }
                .padding(.horizontal)
                .font(.system(.body, design: .monospaced))
                
                Spacer()
                
                Button(action: detectionManager.toggleDetection) {
                    Label(detectionManager.isDetecting ? "Stop Detection" : "Start Detection",
                          systemImage: detectionManager.isDetecting ? "stop.circle" : "play.circle")
                }
                .buttonStyle(.borderedProminent)
                .padding()
            }
            .frame(minWidth: 250, maxWidth: 300)
            .padding()
            
            // Right Panel - Media View
            CameraView(detectionManager: detectionManager)
                .background(Color.black)
        }
        .frame(minWidth: 800, minHeight: 600)
        .onReceive(NotificationCenter.default.publisher(for: .openTextWindow)) { _ in
            openWindow(id: "Text-View")
        }
        .sheet(isPresented: $showingCameraPicker) {
            CameraPickerView(detectionManager: detectionManager)
        }
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [.movie, .quickTimeMovie, .mpeg4Movie, .image, .jpeg, .png, .heic],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    detectionManager.loadMediaFile(url: url)
                }
            case .failure(let error):
                print("File picker error: \(error)")
            }
        }
    }
}

// MARK: - Camera Picker View
struct CameraPickerView: View {
    @ObservedObject var detectionManager: DetectionManager
    @Environment(\.dismiss) private var dismiss
    @State private var availableCameras: [AVCaptureDevice] = []
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Select Camera")
                .font(.title2)
                .fontWeight(.semibold)
            
            List(availableCameras, id: \.uniqueID) { camera in
                HStack {
                    VStack(alignment: .leading) {
                        Text(camera.localizedName)
                            .font(.headline)
                        Text(camera.uniqueID)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    if camera.uniqueID == detectionManager.selectedCameraID {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.blue)
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    detectionManager.selectCamera(camera)
                    dismiss()
                }
            }
            
            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .buttonStyle(.bordered)
                
                Spacer()
            }
            .padding()
        }
        .frame(width: 400, height: 300)
        .padding()
        .onAppear {
            loadAvailableCameras()
        }
    }
    
    private func loadAvailableCameras() {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .externalUnknown],
            mediaType: .video,
            position: .unspecified
        )
        availableCameras = discoverySession.devices
    }
}

// MARK: - Camera View (NSViewRepresentable)
struct CameraView: NSViewRepresentable {
    @ObservedObject var detectionManager: DetectionManager
    
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.black.cgColor
        
        detectionManager.setupMedia(in: view)
        return view
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        // Updates handled by DetectionManager
    }
}

// MARK: - Detection Manager
class DetectionManager: NSObject, ObservableObject {
    // Published properties for UI
    @Published var showCameraFeed = true
    @Published var faceDetectionEnabled = false
    @Published var faceLandmarksEnabled = false
    @Published var handDetectionEnabled = false
    @Published var bodyDetectionEnabled = false
    @Published var objectDetectionEnabled = false
    @Published var textDetectionEnabled = false
    @Published var contourDetectionEnabled = true
    
    @Published var currentFPS: Double = 0
    @Published var detectionCount: Int = 0
    @Published var statusMessage: String = "Ready"
    @Published var isDetecting: Bool = false
    
    // Media source properties
    @Published var mediaSourceType: MediaSourceType = .camera
    @Published var selectedCameraID: String = ""
    @Published var selectedCameraName: String = "Default Camera"
    @Published var isVideoPlaying: Bool = false
    
    // New property for philosophical view
    @Published var currentDetection: String? = nil
    @Published var recentDetectedObjects: Set<String> = []
    
    // Camera and Vision
    private var captureSession: AVCaptureSession?
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated)
    private weak var previewView: NSView?
    
    // Video player - using Timer instead of CVDisplayLink for simplicity
    private var player: AVPlayer?
    private var playerLayer: AVPlayerLayer?
    private var playerItemVideoOutput: AVPlayerItemVideoOutput?
    private var videoProcessingTimer: Timer?
    
    // Image display
    private var imageLayer: CALayer?
    
    // Overlay layers
    private var overlayLayer = CALayer()
    private var detectionLayers: [String: CALayer] = [:]
    
    // Video dimensions for proper coordinate mapping
    private var videoDimensions: CMVideoDimensions?
    private var videoOrientation: CGImagePropertyOrientation = .up
    
    // Vision requests
    private lazy var faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: handleFaceDetection)
    private lazy var faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: handleFaceLandmarks)
    private lazy var handPoseRequest = VNDetectHumanHandPoseRequest(completionHandler: handleHandPose)
    private lazy var bodyPoseRequest = VNDetectHumanBodyPoseRequest(completionHandler: handleBodyPose)
    private lazy var textDetectionRequest = VNRecognizeTextRequest(completionHandler: handleTextDetection)
    private lazy var contoursRequest = VNDetectContoursRequest(completionHandler: handleContours)
    
    // YOLO models
    private var yoloRequest: VNCoreMLRequest?
    
    // Performance tracking
    private var frameCount = 0
    private var fpsTimer: Timer?
    private var lastFrameTime = CACurrentMediaTime()
    
    // Detection history for recent objects
    private var detectionHistory: [(String, Date)] = []
    
    override init() {
        super.init()
        setupVisionRequests()
        setupYOLOModels()
        startFPSTimer()
        setupDefaultCamera()
    }
    
    deinit {
        cleanup()
    }
    
    // MARK: - Setup Methods
    
    private func setupDefaultCamera() {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .externalUnknown],
            mediaType: .video,
            position: .unspecified
        )
        
        if let defaultCamera = discoverySession.devices.first {
            selectedCameraID = defaultCamera.uniqueID
            selectedCameraName = defaultCamera.localizedName
        }
    }
    
    func setupMedia(in view: NSView) {
        previewView = view
        
        // Setup overlay layer
        overlayLayer.frame = view.bounds
        overlayLayer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        view.layer?.addSublayer(overlayLayer)
        
        // Create detection layers
        let layerNames = ["face", "landmarks", "hand", "body", "object", "text", "contour"]
        for name in layerNames {
            let layer = CALayer()
            layer.frame = view.bounds
            layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            detectionLayers[name] = layer
            overlayLayer.addSublayer(layer)
        }
        
        switch mediaSourceType {
        case .camera:
            setupCamera(in: view)
        case .video:
            setupVideoPlayer(in: view)
        case .image:
            setupImageDisplay(in: view)
        }
    }
    
    func setupCamera(in view: NSView) {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .high
        
        // Find the selected camera
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .externalUnknown],
            mediaType: .video,
            position: .unspecified
        )
        
        guard let videoCaptureDevice = discoverySession.devices.first(where: { $0.uniqueID == selectedCameraID }) ??
                discoverySession.devices.first else {
            updateStatus("No camera available")
            return
        }
        
        do {
            let videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
            
            if captureSession?.canAddInput(videoInput) == true {
                captureSession?.addInput(videoInput)
            }
            
            // Get video dimensions
            let formatDescription = videoCaptureDevice.activeFormat.formatDescription
            videoDimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
            
            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput?.alwaysDiscardsLateVideoFrames = true
            videoDataOutput?.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            
            if captureSession?.canAddOutput(videoDataOutput!) == true {
                captureSession?.addOutput(videoDataOutput!)
            }
            
            // Setup preview layer
            videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
            videoPreviewLayer?.videoGravity = .resizeAspect
            videoPreviewLayer?.frame = view.bounds
            videoPreviewLayer?.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            
            view.layer?.insertSublayer(videoPreviewLayer!, at: 0)
            
            // Start camera session
            DispatchQueue.global(qos: .background).async { [weak self] in
                self?.captureSession?.startRunning()
                DispatchQueue.main.async {
                    self?.updateStatus("Camera ready")
                }
            }
            
        } catch {
            updateStatus("Error: \(error.localizedDescription)")
        }
    }
    
    private func setupVideoPlayer(in view: NSView) {
        guard let player = player else { return }
        
        playerLayer = AVPlayerLayer(player: player)
        playerLayer?.videoGravity = .resizeAspect
        playerLayer?.frame = view.bounds
        playerLayer?.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        
        view.layer?.insertSublayer(playerLayer!, at: 0)
        
        // Setup video output for frame processing
        let settings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        playerItemVideoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: settings)
        player.currentItem?.add(playerItemVideoOutput!)
        
        updateStatus("Video ready")
    }
    
    private func setupImageDisplay(in view: NSView) {
        // Image display setup would go here
        updateStatus("Image ready")
    }
    
    private func startVideoProcessingTimer() {
        videoProcessingTimer = Timer.scheduledTimer(withTimeInterval: 1.0/30.0, repeats: true) { [weak self] _ in
            self?.processVideoFrame()
        }
    }
    
    private func stopVideoProcessingTimer() {
        videoProcessingTimer?.invalidate()
        videoProcessingTimer = nil
    }
    
    func selectCamera(_ camera: AVCaptureDevice) {
        selectedCameraID = camera.uniqueID
        selectedCameraName = camera.localizedName
        mediaSourceType = .camera
        
        // Restart media setup with new camera
        if let view = previewView {
            cleanup()
            setupMedia(in: view)
        }
    }
    
    func loadMediaFile(url: URL) {
        let resourceValues = try? url.resourceValues(forKeys: [.contentTypeKey])
        let contentType = resourceValues?.contentType
        
        if contentType?.conforms(to: .movie) == true {
            // Load video
            mediaSourceType = .video
            player = AVPlayer(url: url)
            isVideoPlaying = false
            
            if let view = previewView {
                cleanup()
                setupMedia(in: view)
            }
        } else if contentType?.conforms(to: .image) == true {
            // Load image
            mediaSourceType = .image
            
            // Process single image for detection
            if let nsImage = NSImage(contentsOf: url),
               let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                processImage(cgImage)
            }
            
            if let view = previewView {
                cleanup()
                setupMedia(in: view)
            }
        }
    }
    
    func toggleVideoPlayback() {
        guard let player = player else { return }
        
        if isVideoPlaying {
            player.pause()
            stopVideoProcessingTimer()
        } else {
            player.play()
            if isDetecting {
                startVideoProcessingTimer()
            }
        }
        isVideoPlaying.toggle()
    }
    
    func restartVideo() {
        guard let player = player else { return }
        player.seek(to: .zero)
        if isVideoPlaying {
            player.play()
            if isDetecting {
                startVideoProcessingTimer()
            }
        }
    }
    
    private func processVideoFrame() {
        guard let playerItemVideoOutput = playerItemVideoOutput,
              let player = player,
              isDetecting else { return }
        
        let currentTime = player.currentTime()
        
        if playerItemVideoOutput.hasNewPixelBuffer(forItemTime: currentTime) {
            if let pixelBuffer = playerItemVideoOutput.copyPixelBuffer(forItemTime: currentTime, itemTimeForDisplay: nil) {
                DispatchQueue.main.async {
                    self.processPixelBuffer(pixelBuffer)
                    self.frameCount += 1
                }
            }
        }
    }
    
    private func processImage(_ cgImage: CGImage) {
        guard isDetecting else { return }
        
        let imageRequestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        var requests = [VNRequest]()
        
        // Add requests based on toggle states
        if faceDetectionEnabled {
            requests.append(faceDetectionRequest)
            if faceLandmarksEnabled {
                requests.append(faceLandmarksRequest)
            }
        }
        
        if handDetectionEnabled {
            requests.append(handPoseRequest)
        }
        
        if bodyDetectionEnabled {
            requests.append(bodyPoseRequest)
        }
        
        if objectDetectionEnabled, let yoloRequest = yoloRequest {
            requests.append(yoloRequest)
        }
        
        if textDetectionEnabled {
            requests.append(textDetectionRequest)
        }
        
        if contourDetectionEnabled {
            requests.append(contoursRequest)
        }
        
        if !requests.isEmpty {
            do {
                try imageRequestHandler.perform(requests)
            } catch {
                print("Failed to perform request: \(error)")
            }
        }
    }
    
    private func cleanup() {
        // Stop camera session
        captureSession?.stopRunning()
        captureSession = nil
        
        // Stop video player
        player?.pause()
        stopVideoProcessingTimer()
        
        // Remove layers
        videoPreviewLayer?.removeFromSuperlayer()
        playerLayer?.removeFromSuperlayer()
        imageLayer?.removeFromSuperlayer()
        
        videoPreviewLayer = nil
        playerLayer = nil
        imageLayer = nil
        playerItemVideoOutput = nil
    }
    
    private func setupVisionRequests() {
        // Configure face detection
        faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision3
        faceLandmarksRequest.revision = VNDetectFaceLandmarksRequestRevision3
        
        // Configure hand pose
        handPoseRequest.maximumHandCount = 16
        
        // Configure text detection
        textDetectionRequest.recognitionLevel = .fast
        textDetectionRequest.usesLanguageCorrection = true
        
        // Configure contours
        contoursRequest.contrastAdjustment = 1.0
        contoursRequest.detectsDarkOnLight = true
    }
    
    private func setupYOLOModels() {
        // Load YOLOv3
        if let modelURL = Bundle.main.url(forResource: "YOLOv3TinyFP16", withExtension: "mlmodelc") {
            do {
                let model = try MLModel(contentsOf: modelURL)
                let visionModel = try VNCoreMLModel(for: model)
                yoloRequest = VNCoreMLRequest(model: visionModel, completionHandler: handleObjectDetection)
                yoloRequest?.imageCropAndScaleOption = .scaleFit
                updateStatus("YOLOv3TinyFP16 loaded")
            } catch {
                print("Error loading YOLOv3TinyFP16: \(error)")
            }
        }
    }
    
    // MARK: - Camera Control
    
    func updateCameraVisibility() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.videoPreviewLayer?.isHidden = !self.showCameraFeed
            self.playerLayer?.isHidden = !self.showCameraFeed
            self.imageLayer?.isHidden = !self.showCameraFeed
        }
    }
    
    func toggleDetection() {
        isDetecting.toggle()
        updateStatus(isDetecting ? "Detection started" : "Detection stopped")
        
        if !isDetecting {
            // Clear all detection layers when stopping
            for (name, _) in detectionLayers {
                clearLayer(name)
            }
            // Clear current detection
            DispatchQueue.main.async {
                self.currentDetection = nil
                self.recentDetectedObjects = []
                self.detectionHistory = []
            }
            
            // Stop video processing when detection stops
            if mediaSourceType == .video {
                stopVideoProcessingTimer()
            }
        } else {
            // Start video processing when detection starts and video is playing
            if mediaSourceType == .video && isVideoPlaying {
                startVideoProcessingTimer()
            }
        }
    }
    
    func clearLayer(_ name: String) {
        DispatchQueue.main.async { [weak self] in
            guard let layer = self?.detectionLayers[name] else { return }
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            self?.updateDetectionCount()
        }
    }
    
    // MARK: - Pixel Buffer Processing
    
    private func processPixelBuffer(_ pixelBuffer: CVPixelBuffer) {
        var requests = [VNRequest]()
        
        // Add requests based on toggle states
        if faceDetectionEnabled {
            requests.append(faceDetectionRequest)
            if faceLandmarksEnabled {
                requests.append(faceLandmarksRequest)
            }
        }
        
        if handDetectionEnabled {
            requests.append(handPoseRequest)
        }
        
        if bodyDetectionEnabled {
            requests.append(bodyPoseRequest)
        }
        
        if objectDetectionEnabled, let yoloRequest = yoloRequest {
            requests.append(yoloRequest)
        }
        
        if textDetectionEnabled {
            requests.append(textDetectionRequest)
        }
        
        if contourDetectionEnabled {
            requests.append(contoursRequest)
        }
        
        // Perform Vision requests
        if !requests.isEmpty {
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
            
            do {
                try imageRequestHandler.perform(requests)
            } catch {
                print("Failed to perform request: \(error)")
            }
        }
    }
    
    // MARK: - Vision Request Handlers
    
    private func handleFaceDetection(request: VNRequest, error: Error?) {
        guard faceDetectionEnabled else { return }
        guard let results = request.results as? [VNFaceObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["face"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            if !results.isEmpty {
                self.updateCurrentDetection("a face")
            }
            
            for face in results {
                self.drawBoundingBox(face.boundingBox, in: layer, color: .green, label: "Face")
            }
            
            self.updateDetectionCount()
        }
    }
    
    private func handleFaceLandmarks(request: VNRequest, error: Error?) {
        guard faceLandmarksEnabled else { return }
        guard let results = request.results as? [VNFaceObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["landmarks"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            for face in results {
                if let landmarks = face.landmarks {
                    self.drawFaceLandmarks(landmarks, boundingBox: face.boundingBox, in: layer)
                }
            }
        }
    }
    
    private func handleHandPose(request: VNRequest, error: Error?) {
        guard handDetectionEnabled else { return }
        guard let results = request.results as? [VNHumanHandPoseObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["hand"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            if !results.isEmpty {
                self.updateCurrentDetection("a hand")
            }
            
            for hand in results {
                do {
                    let points = try hand.recognizedPoints(.all)
                    self.drawHandSkeleton(points, in: layer)
                } catch {
                    print("Error getting hand points: \(error)")
                }
            }
            
            self.updateDetectionCount()
        }
    }
    
    private func handleBodyPose(request: VNRequest, error: Error?) {
        guard bodyDetectionEnabled else { return }
        guard let results = request.results as? [VNHumanBodyPoseObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["body"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            if !results.isEmpty {
                self.updateCurrentDetection("a body")
            }
            
            for body in results {
                do {
                    let points = try body.recognizedPoints(.all)
                    self.drawBodySkeleton(points, in: layer)
                } catch {
                    print("Error getting body points: \(error)")
                }
            }
            
            self.updateDetectionCount()
        }
    }
    
    private func handleObjectDetection(request: VNRequest, error: Error?) {
        guard objectDetectionEnabled else { return }
        guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["object"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            var detectedObjects: [String] = []
            
            for object in results where object.confidence > 0.1 {
                let label = object.labels.first?.identifier ?? "Unknown"
                let confidence = object.labels.first?.confidence ?? 0
                let text = "\(label): \(String(format: "%.2f", confidence))"
                
                if object.confidence > 0.3 {
                    detectedObjects.append("a \(label)")
                }
                
                self.drawBoundingBox(object.boundingBox, in: layer, color: .systemBlue, label: text)
            }
            
            for obj in detectedObjects {
                self.updateCurrentDetection(obj)
            }
            
            self.updateDetectionCount()
        }
    }
    
    private func handleTextDetection(request: VNRequest, error: Error?) {
        guard textDetectionEnabled else { return }
        guard let results = request.results as? [VNRecognizedTextObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["text"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            if !results.isEmpty {
                self.updateCurrentDetection("text")
            }
            
            for textObservation in results {
                if let topCandidate = textObservation.topCandidates(1).first {
                    self.drawBoundingBox(textObservation.boundingBox, in: layer,
                                       color: .systemYellow, label: topCandidate.string)
                }
            }
            
            self.updateDetectionCount()
        }
    }
    
    private func handleContours(request: VNRequest, error: Error?) {
        guard contourDetectionEnabled else { return }
        guard let result = request.results?.first as? VNContoursObservation else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["contour"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            let path = CGMutablePath()
            
            for contourIndex in 0..<min(result.contourCount, 500) {
                if let contour = try? result.contour(at: contourIndex) {
                    let points = contour.normalizedPoints
                    
                    for (index, point) in points.enumerated() {
                        let cgPoint = self.convertNormalizedPoint(CGPoint(x: CGFloat(point.x), y: CGFloat(point.y)))
                        
                        if index == 0 {
                            path.move(to: cgPoint)
                        } else {
                            path.addLine(to: cgPoint)
                        }
                    }
                }
            }
            
            let shapeLayer = CAShapeLayer()
            shapeLayer.path = path
            shapeLayer.fillColor = NSColor.clear.cgColor
            shapeLayer.strokeColor = NSColor.systemCyan.cgColor
            shapeLayer.lineWidth = 3.0
            layer.addSublayer(shapeLayer)
            
            self.updateDetectionCount()
        }
    }
    
    // MARK: - Update Current Detection
    
    private func updateCurrentDetection(_ detection: String) {
        DispatchQueue.main.async {
            let now = Date()
            self.detectionHistory = self.detectionHistory.filter { now.timeIntervalSince($0.1) <= 1.0 }
            self.detectionHistory.append((detection, now))
            self.recentDetectedObjects = Set(self.detectionHistory.map { $0.0 })
        }
    }
    
    // MARK: - Coordinate Conversion
    
    private func convertNormalizedRect(_ rect: CGRect) -> CGRect {
        guard let previewView = previewView else { return .zero }
        
        let viewFrame = previewView.bounds
        
        // For video player or image display, use the full view bounds
        if mediaSourceType == .video || mediaSourceType == .image {
            return CGRect(
                x: rect.minX * viewFrame.width,
                y: (1 - rect.maxY) * viewFrame.height,
                width: rect.width * viewFrame.width,
                height: rect.height * viewFrame.height
            )
        }
        
        // For camera, use the preview layer conversion
        guard let previewLayer = videoPreviewLayer else { return .zero }
        
        let videoRect = previewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        return CGRect(
            x: videoRect.minX + rect.minX * videoRect.width,
            y: videoRect.minY + rect.minY * videoRect.height,
            width: rect.width * videoRect.width,
            height: rect.height * videoRect.height
        )
    }
    
    private func convertNormalizedPoint(_ point: CGPoint) -> CGPoint {
        guard let previewView = previewView else { return .zero }
        
        let viewFrame = previewView.bounds
        
        // For video player or image display, use the full view bounds
        if mediaSourceType == .video || mediaSourceType == .image {
            return CGPoint(
                x: point.x * viewFrame.width,
                y: (1 - point.y) * viewFrame.height
            )
        }
        
        // For camera, use the preview layer conversion
        guard let previewLayer = videoPreviewLayer else { return .zero }
        
        let videoRect = previewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        return CGPoint(
            x: videoRect.minX + point.x * videoRect.width,
            y: videoRect.minY + point.y * videoRect.height
        )
    }
    
    private func convertNormalizedPoint(_ point: CGPoint, in rect: CGRect) -> CGPoint {
        guard let previewView = previewView else { return .zero }
        
        let viewFrame = previewView.bounds
        let convertedRect = convertNormalizedRect(rect)
        
        return CGPoint(
            x: convertedRect.minX + point.x * convertedRect.width,
            y: convertedRect.minY + point.y * convertedRect.height
        )
    }
    
    // MARK: - Drawing Methods
    
    private func drawBoundingBox(_ rect: CGRect, in layer: CALayer, color: NSColor, label: String? = nil) {
        let convertedRect = convertNormalizedRect(rect)
        
        let box = CALayer()
        box.frame = convertedRect
        box.borderColor = color.cgColor
        box.borderWidth = 2.0
        box.cornerRadius = 4.0
        box.backgroundColor = color.withAlphaComponent(0.1).cgColor
        layer.addSublayer(box)
        
        if let label = label {
            let textLayer = CATextLayer()
            textLayer.string = label
            textLayer.fontSize = 12
            textLayer.foregroundColor = NSColor.white.cgColor
            textLayer.backgroundColor = color.withAlphaComponent(0.7).cgColor
            textLayer.alignmentMode = .center
            textLayer.contentsScale = 2.0
            
            let size = label.size(withAttributes: [.font: NSFont.systemFont(ofSize: 12)])
            textLayer.frame = CGRect(x: convertedRect.minX, y: convertedRect.minY - size.height - 2,
                                    width: size.width + 10, height: size.height + 4)
            layer.addSublayer(textLayer)
        }
    }
    
    private func drawFaceLandmarks(_ landmarks: VNFaceLandmarks2D, boundingBox: CGRect, in layer: CALayer) {
        let features = [
            landmarks.leftEye,
            landmarks.rightEye,
            landmarks.leftEyebrow,
            landmarks.rightEyebrow,
            landmarks.nose,
            landmarks.outerLips,
            landmarks.innerLips
        ]
        
        for feature in features.compactMap({ $0 }) {
            let path = CGMutablePath()
            let points = feature.normalizedPoints
            
            for (index, point) in points.enumerated() {
                let normalizedPoint = CGPoint(x: CGFloat(point.x), y: CGFloat(point.y))
                let cgPoint = convertNormalizedPoint(normalizedPoint, in: boundingBox)
                
                if index == 0 {
                    path.move(to: cgPoint)
                } else {
                    path.addLine(to: cgPoint)
                }
            }
            
            let shapeLayer = CAShapeLayer()
            shapeLayer.path = path
            shapeLayer.fillColor = NSColor.clear.cgColor
            shapeLayer.strokeColor = NSColor.systemRed.cgColor
            shapeLayer.lineWidth = 1.0
            layer.addSublayer(shapeLayer)
        }
    }
    
    private func drawHandSkeleton(_ points: [VNHumanHandPoseObservation.JointName : VNRecognizedPoint], in layer: CALayer) {
        // Draw joints
        for (_, point) in points where point.confidence > 0.3 {
            let cgPoint = convertNormalizedPoint(CGPoint(x: point.location.x, y: point.location.y))
            
            let circle = CALayer()
            circle.frame = CGRect(x: cgPoint.x - 3, y: cgPoint.y - 3, width: 6, height: 6)
            circle.backgroundColor = NSColor.systemOrange.cgColor
            circle.cornerRadius = 3
            layer.addSublayer(circle)
        }
        
        // Draw connections
        let connections: [(VNHumanHandPoseObservation.JointName, VNHumanHandPoseObservation.JointName)] = [
            (.wrist, .thumbCMC), (.thumbCMC, .thumbMP), (.thumbMP, .thumbIP), (.thumbIP, .thumbTip),
            (.wrist, .indexMCP), (.indexMCP, .indexPIP), (.indexPIP, .indexDIP), (.indexDIP, .indexTip),
            (.wrist, .middleMCP), (.middleMCP, .middlePIP), (.middlePIP, .middleDIP), (.middleDIP, .middleTip),
            (.wrist, .ringMCP), (.ringMCP, .ringPIP), (.ringPIP, .ringDIP), (.ringDIP, .ringTip),
            (.wrist, .littleMCP), (.littleMCP, .littlePIP), (.littlePIP, .littleDIP), (.littleDIP, .littleTip)
        ]
        
        for (joint1, joint2) in connections {
            if let point1 = points[joint1], let point2 = points[joint2],
               point1.confidence > 0.3 && point2.confidence > 0.3 {
                
                let path = CGMutablePath()
                path.move(to: convertNormalizedPoint(CGPoint(x: point1.location.x, y: point1.location.y)))
                path.addLine(to: convertNormalizedPoint(CGPoint(x: point2.location.x, y: point2.location.y)))
                
                let line = CAShapeLayer()
                line.path = path
                line.strokeColor = NSColor.systemOrange.cgColor
                line.lineWidth = 1.5
                layer.addSublayer(line)
            }
        }
    }
    
    private func drawBodySkeleton(_ points: [VNHumanBodyPoseObservation.JointName : VNRecognizedPoint], in layer: CALayer) {
        // Draw joints
        for (_, point) in points where point.confidence > 0.3 {
            let cgPoint = convertNormalizedPoint(CGPoint(x: point.location.x, y: point.location.y))
            
            let circle = CALayer()
            circle.frame = CGRect(x: cgPoint.x - 4, y: cgPoint.y - 4, width: 8, height: 8)
            circle.backgroundColor = NSColor.systemPurple.cgColor
            circle.cornerRadius = 4
            layer.addSublayer(circle)
        }
        
        // Draw skeleton
        let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
            (.nose, .neck), (.neck, .leftShoulder), (.neck, .rightShoulder),
            (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
            (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
            (.neck, .root), (.root, .leftHip), (.root, .rightHip),
            (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
            (.rightHip, .rightKnee), (.rightKnee, .rightAnkle)
        ]
        
        for (joint1, joint2) in connections {
            if let point1 = points[joint1], let point2 = points[joint2],
               point1.confidence > 0.3 && point2.confidence > 0.3 {
                
                let path = CGMutablePath()
                path.move(to: convertNormalizedPoint(CGPoint(x: point1.location.x, y: point1.location.y)))
                path.addLine(to: convertNormalizedPoint(CGPoint(x: point2.location.x, y: point2.location.y)))
                
                let line = CAShapeLayer()
                line.path = path
                line.strokeColor = NSColor.systemPurple.cgColor
                line.lineWidth = 2.0
                layer.addSublayer(line)
            }
        }
    }
    
    // MARK: - Utility Methods
    
    private func updateStatus(_ message: String) {
        DispatchQueue.main.async {
            self.statusMessage = message
        }
    }
    
    private func updateDetectionCount() {
        var count = 0
        for layer in detectionLayers.values {
            count += layer.sublayers?.count ?? 0
        }
        DispatchQueue.main.async {
            self.detectionCount = count
        }
    }
    
    private func startFPSTimer() {
        fpsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.currentFPS = Double(self.frameCount)
                self.frameCount = 0
            }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension DetectionManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard isDetecting else { return }
        
        frameCount += 1
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        processPixelBuffer(pixelBuffer)
    }
}
