import SwiftUI
import AppKit
import AVFoundation
import Vision
import CoreML
import UniformTypeIdentifiers

// MARK: - Main App
@main
struct objectdetectionApp: App {
    @StateObject private var detectionManager = DetectionManager()
    @StateObject private var windowManager = WindowManager()

    init() {
        NSWindow.allowsAutomaticWindowTabbing = false
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(detectionManager)
                .environmentObject(windowManager)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1024, height: 768)

        WindowGroup("Machine Perception", id: "Text-View") {
            TextView()
                .environmentObject(detectionManager)
                .onAppear {
                    windowManager.isTextViewOpen = true
                }
                .onDisappear {
                    windowManager.isTextViewOpen = false
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 800, height: 600)

        .commands {
            CommandGroup(after: .toolbar) {
                Button(detectionManager.showSelectionGUI ? "Hide Selection GUI" : "Show Selection GUI") {
                    detectionManager.showSelectionGUI.toggle()
                }
                .keyboardShortcut("v", modifiers: [.command, .option])
            }
        }
    }
}

// MARK: - Window Manager
class WindowManager: ObservableObject {
    @Published var isTextViewOpen: Bool = false
    
    func openTextWindow() {
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

// MARK: - Media Feed Mode
enum MediaFeedMode: String, CaseIterable, Hashable {
    case on
    case off
    case alternating
}

// MARK: - text View
struct TextView: View {
    @EnvironmentObject private var detectionManager: DetectionManager
    @State private var currentObject: String? = nil
    @State private var displayedNegation: String? = nil
    
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
            "have no heart to race",
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
            "cannot have preferences",
            
            // Sensation
            "taste no sweetness",
            "feel no texture",
            "hear no music",
            "see no beauty",
            "smell no flowers",
            "sense no comfort",
            "can'tt experience subjectivity",
            
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
            "cannot empathise",
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
            Color.black
                .ignoresSafeArea()
            
            VStack(spacing: 40) {
                if detectionManager.recentDetectedObjects.isEmpty {
                    Text("I see nothing")
                        .font(.system(size: 72, weight: .regular, design: .monospaced))
                        .foregroundColor(.white.opacity(0.9))
                        .transition(.opacity)
                } else {
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
        .onChange(of: detectionManager.recentDetectedObjects) { _ in
            handleDetectionState()
        }
        .onChange(of: detectionManager.isSpeaking) { speaking in
            if !speaking && !detectionManager.isWaitingAfterSpeech {
                scheduleNextPhrase()
            }
        }
        .onChange(of: detectionManager.isWaitingAfterSpeech) { waiting in
            if !waiting && !detectionManager.isSpeaking {
                generateAndSpeakPhrase()
            }
        }
    }
    
    private func handleDetectionState() {
        if !detectionManager.isSpeaking && !detectionManager.isWaitingAfterSpeech {
            generateAndSpeakPhrase()
        }
    }
    
    private func scheduleNextPhrase() {
        // Wait before next phrase
        let waitTime = detectionManager.recentDetectedObjects.isEmpty ? 5.0 : 2.0
        detectionManager.startWaitingAfterSpeech(duration: waitTime)
    }
    
    private func generateAndSpeakPhrase() {
        if detectionManager.recentDetectedObjects.isEmpty {
            withAnimation {
                currentObject = nil
                displayedNegation = nil
            }
            speakCurrentPhrase()
        } else {
            withAnimation(.easeInOut(duration: 0.3)) {
                currentObject = detectionManager.recentDetectedObjects.randomElement()
                displayedNegation = negations.randomElement() ?? "do not feel"
            }
            speakCurrentPhrase()
        }
    }
    
    private func speakCurrentPhrase() {
        let phrase: String
        if let obj = currentObject, let neg = displayedNegation, !obj.isEmpty {
            phrase = "I see \(obj.lowercased()). But I \(neg)."
        } else {
            phrase = "I see nothing"
        }
        detectionManager.speak(text: phrase)
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
        for (keywords, color) in colorCategories {
            if keywords.contains(where: { lowercased.contains($0) }) {
                return color
            }
        }
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
            if detectionManager.showSelectionGUI {
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

                                Button(detectionManager.isMuted ? "Unmute" : "Mute") {
                                    detectionManager.toggleMute()
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
                        Picker("Media Feed", selection: $detectionManager.mediaFeedMode) {
                            Text("On").tag(MediaFeedMode.on)
                            Text("Off").tag(MediaFeedMode.off)
                            Text("Alternate (10s)").tag(MediaFeedMode.alternating)
                        }
                        .pickerStyle(.menu)
                        .help("Choose how the media feed is shown. Alternate toggles every 10 seconds.")
                        
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
                    
                    Divider()
                    
                    Button(action: {
                        if windowManager.isTextViewOpen {
                            // Close window via notification
                            NotificationCenter.default.post(name: .closeTextWindow, object: nil)
                        } else {
                            windowManager.openTextWindow()
                        }
                    }) {
                        Label(windowManager.isTextViewOpen ? "Close Philosophy View" : "Open Philosophy View",
                              systemImage: windowManager.isTextViewOpen ? "eye.slash" : "eye")
                    }
                    .buttonStyle(.bordered)
                    .padding(.horizontal)
                    
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
            }
            
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

extension Notification.Name {
    static let closeTextWindow = Notification.Name("closeTextWindow")
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
    @Published var showCameraFeed = true
    @Published var faceDetectionEnabled = true
    @Published var faceLandmarksEnabled = true
    @Published var handDetectionEnabled = true
    @Published var bodyDetectionEnabled = true
    @Published var objectDetectionEnabled = true
    @Published var textDetectionEnabled = false
    @Published var contourDetectionEnabled = true
    @Published var showSelectionGUI: Bool = true
    @Published var mediaFeedMode: MediaFeedMode = .on {
        didSet { applyMediaFeedMode() }
    }
    
    @Published var currentFPS: Double = 0
    @Published var detectionCount: Int = 0
    @Published var statusMessage: String = "Ready"
    @Published var isDetecting: Bool = false
    
    @Published var isSpeaking: Bool = false
    @Published var isWaitingAfterSpeech: Bool = false
    private let speechSynth = AVSpeechSynthesizer()
    private var preferredVoice: AVSpeechSynthesisVoice? = nil
    private var speechTimeoutTimer: Timer?
    private var waitTimer: Timer?
    private var mediaFeedTimer: Timer?
    
    @Published var mediaSourceType: MediaSourceType = .camera
    @Published var selectedCameraID: String = ""
    @Published var selectedCameraName: String = "Default Camera"
    @Published var isVideoPlaying: Bool = false
    @Published var isMuted: Bool = false
    
    @Published var currentDetection: String? = nil
    @Published var recentDetectedObjects: Set<String> = []
    
    private var captureSession: AVCaptureSession?
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated)
    private weak var previewView: NSView?
    
    private var player: AVPlayer?
    private var playerLayer: AVPlayerLayer?
    private var playerItemVideoOutput: AVPlayerItemVideoOutput?
    private var videoProcessingTimer: Timer?
    private var displayLink: CVDisplayLink?
    
    private var imageLayer: CALayer?
    
    private var overlayLayer = CALayer()
    private var detectionLayers: [String: CALayer] = [:]
    
    private var videoDimensions: CMVideoDimensions?
    private var videoOrientation: CGImagePropertyOrientation = .up
    
    private lazy var faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: handleFaceDetection)
    private lazy var faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: handleFaceLandmarks)
    private lazy var handPoseRequest = VNDetectHumanHandPoseRequest(completionHandler: handleHandPose)
    private lazy var bodyPoseRequest = VNDetectHumanBodyPoseRequest(completionHandler: handleBodyPose)
    private lazy var textDetectionRequest = VNRecognizeTextRequest(completionHandler: handleTextDetection)
    private lazy var contoursRequest = VNDetectContoursRequest(completionHandler: handleContours)
    
    private var yoloRequest: VNCoreMLRequest?
    
    private var frameCount = 0
    private var fpsTimer: Timer?
    private var lastFrameTime = CACurrentMediaTime()
    
    private var detectionHistory: [(String, Date)] = []
    private var detectionDecayTimer: Timer?
    private let detectionDecayInterval: TimeInterval = 2.5
    
    override init() {
        super.init()
        setupVisionRequests()
        setupYOLOModels()
        startFPSTimer()
        setupDefaultCamera()
        startDetectionDecayTimer()
        applyMediaFeedMode()
        
        speechSynth.delegate = self
        preferredVoice = AVSpeechSynthesisVoice.speechVoices().first(where: { $0.name.localizedCaseInsensitiveContains("Ralph") })
            ?? AVSpeechSynthesisVoice(language: "en-US")
    }
    
    deinit {
        cleanup()
        detectionDecayTimer?.invalidate()
        speechTimeoutTimer?.invalidate()
        waitTimer?.invalidate()
        mediaFeedTimer?.invalidate()
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
        
        overlayLayer.frame = view.bounds
        overlayLayer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        overlayLayer.backgroundColor = NSColor.clear.cgColor
        overlayLayer.isOpaque = false
        
        let layerNames = ["face", "landmarks", "hand", "body", "object", "text", "contour"]
        for name in layerNames {
            let layer = CALayer()
            layer.frame = view.bounds
            layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            layer.backgroundColor = NSColor.clear.cgColor
            layer.isOpaque = false
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
        
        // Add overlay layer AFTER media layer to ensure it's on top
        view.layer?.addSublayer(overlayLayer)
    }
    
    func setupCamera(in view: NSView) {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .high
        
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
            
            let formatDescription = videoCaptureDevice.activeFormat.formatDescription
            videoDimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
            
            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput?.alwaysDiscardsLateVideoFrames = true
            videoDataOutput?.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            
            if captureSession?.canAddOutput(videoDataOutput!) == true {
                captureSession?.addOutput(videoDataOutput!)
            }
            
            videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
            videoPreviewLayer?.videoGravity = .resizeAspect
            videoPreviewLayer?.frame = view.bounds
            videoPreviewLayer?.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            
            view.layer?.insertSublayer(videoPreviewLayer!, at: 0)
            
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
        
        let settings: [String: Any] = [
            String(kCVPixelBufferPixelFormatTypeKey): Int(kCVPixelFormatType_32BGRA)
        ]
        
        playerItemVideoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: settings)
        
        if let currentItem = player.currentItem {
            currentItem.add(playerItemVideoOutput!)
            
            // Get video dimensions from the actual video track
            if let track = currentItem.asset.tracks(withMediaType: .video).first {
                let size = track.naturalSize.applying(track.preferredTransform)
                videoDimensions = CMVideoDimensions(width: Int32(abs(size.width)), height: Int32(abs(size.height)))
            }
        }
        
        NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: player.currentItem,
            queue: .main
        ) { [weak self] _ in
            self?.player?.seek(to: .zero)
            if self?.isVideoPlaying == true {
                self?.player?.play()
            }
        }
        
        updateStatus("Video ready")
    }
    
    private func setupImageDisplay(in view: NSView) {
        updateStatus("Image ready")
    }
    
    private func startVideoProcessingTimer() {
        videoProcessingTimer?.invalidate()
        videoProcessingTimer = Timer.scheduledTimer(withTimeInterval: 1.0/15.0, repeats: true) { [weak self] _ in
            self?.processVideoFrame()
        }
        RunLoop.main.add(videoProcessingTimer!, forMode: .common)
    }
    
    private func stopVideoProcessingTimer() {
        videoProcessingTimer?.invalidate()
        videoProcessingTimer = nil
    }
    
    func selectCamera(_ camera: AVCaptureDevice) {
        selectedCameraID = camera.uniqueID
        selectedCameraName = camera.localizedName
        mediaSourceType = .camera
        
        if let view = previewView {
            clearAllDetections()
            cleanup()
            setupMedia(in: view)
        }
    }
    
    func loadMediaFile(url: URL) {
        // Secure scoped access for sandboxed apps
        guard url.startAccessingSecurityScopedResource() else {
            print("Failed to access file")
            return
        }
        defer { url.stopAccessingSecurityScopedResource() }
        
        let resourceValues = try? url.resourceValues(forKeys: [.contentTypeKey])
        let contentType = resourceValues?.contentType
        
        clearAllDetections()
        
        if contentType?.conforms(to: .movie) == true {
            mediaSourceType = .video
            
            let asset = AVAsset(url: url)
            let playerItem = AVPlayerItem(asset: asset)
            player = AVPlayer(playerItem: playerItem)
            isVideoPlaying = false
            
            if let view = previewView {
                cleanup()
                setupMedia(in: view)
            }
            
            updateStatus("Video loaded")
        } else if contentType?.conforms(to: .image) == true {
            mediaSourceType = .image
            
            if let nsImage = NSImage(contentsOf: url),
               let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                
                // Store image dimensions
                videoDimensions = CMVideoDimensions(width: Int32(cgImage.width), height: Int32(cgImage.height))
                
                if let view = previewView {
                    cleanup()
                    setupMedia(in: view)
                    
                    // Display the image
                    imageLayer = CALayer()
                    imageLayer?.contents = cgImage
                    imageLayer?.frame = view.bounds
                    imageLayer?.contentsGravity = .resizeAspect
                    view.layer?.insertSublayer(imageLayer!, at: 0)
                }
                
                processImage(cgImage)
            }
            
            updateStatus("Image loaded")
        }
    }
    
    func toggleMute() {
        isMuted.toggle()
        player?.isMuted = isMuted
        updateStatus(isMuted ? "Muted" : "Unmuted")
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
        updateStatus(isVideoPlaying ? "Video playing" : "Video paused")
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
        
        // Check if there's a new frame available
        guard playerItemVideoOutput.hasNewPixelBuffer(forItemTime: currentTime),
              let pixelBuffer = playerItemVideoOutput.copyPixelBuffer(
                forItemTime: currentTime,
                itemTimeForDisplay: nil
              ) else { return }
        
        // Process the pixel buffer
        self.processPixelBuffer(pixelBuffer)
        self.frameCount += 1
    }
    
    private func processImage(_ cgImage: CGImage) {
        guard isDetecting else { return }
        
        let imageRequestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        var requests = [VNRequest]()
        
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
        captureSession?.stopRunning()
        captureSession = nil
        
        player?.pause()
        stopVideoProcessingTimer()
        
        videoPreviewLayer?.removeFromSuperlayer()
        playerLayer?.removeFromSuperlayer()
        imageLayer?.removeFromSuperlayer()
        overlayLayer.removeFromSuperlayer()
        
        videoPreviewLayer = nil
        playerLayer = nil
        imageLayer = nil
        playerItemVideoOutput = nil
        
        clearAllDetectionLayers()
        
        // Recreate overlay layer for next setup
        overlayLayer = CALayer()
        detectionLayers.removeAll()
    }
    
    private func clearAllDetectionLayers() {
        for (name, _) in detectionLayers {
            clearLayer(name)
        }
    }
    
    private func clearAllDetections() {
        DispatchQueue.main.async {
            self.clearAllDetectionLayers()
            self.detectionHistory = []
            self.recentDetectedObjects = []
            self.currentDetection = nil
        }
    }
    
    private func setupVisionRequests() {
        faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision3
        faceLandmarksRequest.revision = VNDetectFaceLandmarksRequestRevision3
        handPoseRequest.maximumHandCount = 16
        textDetectionRequest.recognitionLevel = .fast
        textDetectionRequest.usesLanguageCorrection = true
        contoursRequest.contrastAdjustment = 1.0
        contoursRequest.detectsDarkOnLight = true
    }
    
    private func setupYOLOModels() {
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
    
    // MARK: - Detection Decay Timer
    
    private func startDetectionDecayTimer() {
        detectionDecayTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.updateDetectionHistory()
        }
    }
    
    private func updateDetectionHistory() {
        let now = Date()
        detectionHistory = detectionHistory.filter { now.timeIntervalSince($0.1) <= detectionDecayInterval }
        
        let newSet = Set(detectionHistory.map { $0.0 })
        if newSet != recentDetectedObjects {
            DispatchQueue.main.async {
                self.recentDetectedObjects = newSet
            }
        }
    }
    
    // MARK: - TTS
    
    func speak(text textToSpeak: String) {
        if isSpeaking || speechSynth.isSpeaking { return }
        
        DispatchQueue.main.async {
            self.isSpeaking = true
        }
        
        let utterance = AVSpeechUtterance(string: textToSpeak)
        utterance.voice = preferredVoice ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 0.9
        
        speechSynth.speak(utterance)
        
        // Timeout safety in case delegate doesn't fire
        speechTimeoutTimer?.invalidate()
        speechTimeoutTimer = Timer.scheduledTimer(withTimeInterval: 15.0, repeats: false) { [weak self] _ in
            if self?.isSpeaking == true {
                self?.speechSynth.stopSpeaking(at: .immediate)
                DispatchQueue.main.async {
                    self?.isSpeaking = false
                }
            }
        }
    }
    
    func stopSpeaking() {
        speechTimeoutTimer?.invalidate()
        if speechSynth.isSpeaking {
            speechSynth.stopSpeaking(at: .immediate)
        }
        DispatchQueue.main.async {
            self.isSpeaking = false
        }
    }
    
    func startWaitingAfterSpeech(duration: TimeInterval) {
        DispatchQueue.main.async {
            self.isWaitingAfterSpeech = true
        }
        
        waitTimer?.invalidate()
        waitTimer = Timer.scheduledTimer(withTimeInterval: duration, repeats: false) { [weak self] _ in
            DispatchQueue.main.async {
                self?.isWaitingAfterSpeech = false
            }
        }
    }
    
    // MARK: - Media Feed Mode Control
    private func applyMediaFeedMode() {
        // Stop any existing timer
        mediaFeedTimer?.invalidate()
        mediaFeedTimer = nil

        switch mediaFeedMode {
        case .on:
            showCameraFeed = true
            updateCameraVisibility()
        case .off:
            showCameraFeed = false
            updateCameraVisibility()
        case .alternating:
            // Start alternating every 10 seconds
            startMediaFeedAlternatingTimer()
        }
    }
    
    private func startMediaFeedAlternatingTimer() {
        mediaFeedTimer?.invalidate()

        // Start with showing the feed
        showCameraFeed = true
        updateCameraVisibility()

        mediaFeedTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.showCameraFeed.toggle()
            self.updateCameraVisibility()
        }
        if let timer = mediaFeedTimer {
            RunLoop.main.add(timer, forMode: .common)
        }
    }
    
    // MARK: - Camera Control
    
    func updateCameraVisibility() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.videoPreviewLayer?.opacity = self.showCameraFeed ? 1.0 : 0.0
            self.playerLayer?.opacity = self.showCameraFeed ? 1.0 : 0.0
            self.imageLayer?.opacity = self.showCameraFeed ? 1.0 : 0.0
        }
    }
    
    func toggleDetection() {
        isDetecting.toggle()
        updateStatus(isDetecting ? "Detection started" : "Detection stopped")
        
        if !isDetecting {
            clearAllDetections()
            stopSpeaking()
            waitTimer?.invalidate()
            isWaitingAfterSpeech = false
            
            if mediaSourceType == .video {
                stopVideoProcessingTimer()
            }
        } else {
            // Start detection based on media type
            if mediaSourceType == .video && isVideoPlaying {
                startVideoProcessingTimer()
                updateStatus("Detecting on video")
            } else if mediaSourceType == .camera {
                updateStatus("Detecting on camera")
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
            shapeLayer.lineWidth = 2.0
            shapeLayer.opacity = 0.8
            layer.addSublayer(shapeLayer)
            
            self.updateDetectionCount()
        }
    }
    
    // MARK: - Update Current Detection
    
    private func updateCurrentDetection(_ detection: String) {
        let now = Date()
        detectionHistory.append((detection, now))
    }
    
    // MARK: - Coordinate Conversion
    
    private func convertNormalizedRect(_ rect: CGRect) -> CGRect {
        guard let previewView = previewView else { return .zero }
        
        // For video player or image display, calculate based on actual media resolution
        if mediaSourceType == .video || mediaSourceType == .image {
            guard let videoDimensions = videoDimensions else { return .zero }
            
            let mediaSize = CGSize(width: CGFloat(videoDimensions.width), height: CGFloat(videoDimensions.height))
            
            // Calculate the rect where media is actually displayed (respecting aspect ratio)
            let videoRect = AVMakeRect(aspectRatio: mediaSize, insideRect: previewView.bounds)
            
            return CGRect(
                x: videoRect.minX + rect.minX * videoRect.width,
                y: videoRect.minY + rect.minY * videoRect.height,
                width: rect.width * videoRect.width,
                height: rect.height * videoRect.height
            )
        }
        
        // For camera, use the preview layer conversion which handles aspect ratio
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
        
        // For video player or image display, calculate based on actual media resolution
        if mediaSourceType == .video || mediaSourceType == .image {
            guard let videoDimensions = videoDimensions else { return .zero }
            
            let mediaSize = CGSize(width: CGFloat(videoDimensions.width), height: CGFloat(videoDimensions.height))
            
            // Calculate the rect where media is actually displayed (respecting aspect ratio)
            let videoRect = AVMakeRect(aspectRatio: mediaSize, insideRect: previewView.bounds)
            
            return CGPoint(
                x: videoRect.minX + point.x * videoRect.width,
                y: videoRect.minY + point.y * videoRect.height
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
        box.borderWidth = 3.0
        box.cornerRadius = 4.0
        box.backgroundColor = color.withAlphaComponent(0.2).cgColor
        layer.addSublayer(box)
        
        if let label = label {
            let textLayer = CATextLayer()
            textLayer.string = label
            textLayer.fontSize = 14
            textLayer.foregroundColor = NSColor.white.cgColor
            textLayer.backgroundColor = color.withAlphaComponent(0.8).cgColor
            textLayer.alignmentMode = .center
            textLayer.contentsScale = NSScreen.main?.backingScaleFactor ?? 2.0
            
            let size = label.size(withAttributes: [.font: NSFont.systemFont(ofSize: 14)])
            textLayer.frame = CGRect(x: convertedRect.minX, y: convertedRect.minY - size.height - 4,
                                    width: size.width + 12, height: size.height + 6)
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
            shapeLayer.lineWidth = 2.0
            layer.addSublayer(shapeLayer)
        }
    }
    
    private func drawHandSkeleton(_ points: [VNHumanHandPoseObservation.JointName : VNRecognizedPoint], in layer: CALayer) {
        for (_, point) in points where point.confidence > 0.3 {
            let cgPoint = convertNormalizedPoint(CGPoint(x: point.location.x, y: point.location.y))
            
            let circle = CALayer()
            circle.frame = CGRect(x: cgPoint.x - 4, y: cgPoint.y - 4, width: 8, height: 8)
            circle.backgroundColor = NSColor.systemOrange.cgColor
            circle.cornerRadius = 4
            circle.borderColor = NSColor.white.cgColor
            circle.borderWidth = 1
            layer.addSublayer(circle)
        }
        
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
                line.lineWidth = 2.5
                layer.addSublayer(line)
            }
        }
    }
    
    private func drawBodySkeleton(_ points: [VNHumanBodyPoseObservation.JointName : VNRecognizedPoint], in layer: CALayer) {
        for (_, point) in points where point.confidence > 0.3 {
            let cgPoint = convertNormalizedPoint(CGPoint(x: point.location.x, y: point.location.y))
            
            let circle = CALayer()
            circle.frame = CGRect(x: cgPoint.x - 5, y: cgPoint.y - 5, width: 10, height: 10)
            circle.backgroundColor = NSColor.systemPurple.cgColor
            circle.cornerRadius = 5
            circle.borderColor = NSColor.white.cgColor
            circle.borderWidth = 1
            layer.addSublayer(circle)
        }
        
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
                line.lineWidth = 3.0
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

// MARK: - AVSpeechSynthesizerDelegate
extension DetectionManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        speechTimeoutTimer?.invalidate()
        DispatchQueue.main.async { [weak self] in
            self?.isSpeaking = false
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        speechTimeoutTimer?.invalidate()
        DispatchQueue.main.async { [weak self] in
            self?.isSpeaking = false
        }
    }
}
