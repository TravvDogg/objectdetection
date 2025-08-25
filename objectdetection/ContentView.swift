import SwiftUI
import AVFoundation
import Vision
import CoreML

// MARK: - SwiftUI View
struct ContentView: View {
    @StateObject private var detectionManager = DetectionManager()
    
    var body: some View {
        HSplitView {
            // Left Panel - Controls
            VStack(alignment: .leading, spacing: 20) {
                Text("Detection Controls")
                    .font(.headline)
                    .padding(.top)
                
                VStack(alignment: .leading, spacing: 10) {
                    Toggle("Face Detection", isOn: $detectionManager.faceDetectionEnabled)
                    Toggle("Face Landmarks", isOn: $detectionManager.faceLandmarksEnabled)
                    Toggle("Hand Pose", isOn: $detectionManager.handDetectionEnabled)
                    Toggle("Body Pose", isOn: $detectionManager.bodyDetectionEnabled)
                    Toggle("Object Detection (YOLO)", isOn: $detectionManager.objectDetectionEnabled)
                    Toggle("Text Recognition", isOn: $detectionManager.textDetectionEnabled)
                    Toggle("Contour Detection", isOn: $detectionManager.contourDetectionEnabled)
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
                
                Button(action: detectionManager.toggleCamera) {
                    Label(detectionManager.isRunning ? "Stop Camera" : "Start Camera",
                          systemImage: detectionManager.isRunning ? "stop.circle" : "play.circle")
                }
                .buttonStyle(.borderedProminent)
                .padding()
            }
            .frame(minWidth: 250, maxWidth: 300)
            .padding()
            
            // Right Panel - Camera View
            CameraView(detectionManager: detectionManager)
                .background(Color.black)
        }
        .frame(minWidth: 800, minHeight: 600)
    }
}

// MARK: - Camera View (NSViewRepresentable)
struct CameraView: NSViewRepresentable {
    @ObservedObject var detectionManager: DetectionManager
    
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.black.cgColor
        
        detectionManager.setupCamera(in: view)
        return view
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        // Updates handled by DetectionManager
    }
}

// MARK: - Detection Manager
class DetectionManager: NSObject, ObservableObject {
    // Published properties for UI
    @Published var faceDetectionEnabled = true
    @Published var faceLandmarksEnabled = false
    @Published var handDetectionEnabled = false
    @Published var bodyDetectionEnabled = false
    @Published var objectDetectionEnabled = true
    @Published var textDetectionEnabled = false
    @Published var contourDetectionEnabled = false
    
    @Published var currentFPS: Double = 0
    @Published var detectionCount: Int = 0
    @Published var statusMessage: String = "Ready"
    @Published var isRunning: Bool = false
    
    // Camera and Vision
    private var captureSession: AVCaptureSession?
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated)
    private weak var previewView: NSView?
    
    // Overlay layers
    private var overlayLayer = CALayer()
    private var detectionLayers: [String: CALayer] = [:]
    
    // Vision requests
    private lazy var faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: handleFaceDetection)
    private lazy var faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: handleFaceLandmarks)
    private lazy var handPoseRequest = VNDetectHumanHandPoseRequest(completionHandler: handleHandPose)
    private lazy var bodyPoseRequest = VNDetectHumanBodyPoseRequest(completionHandler: handleBodyPose)
    private lazy var textDetectionRequest = VNRecognizeTextRequest(completionHandler: handleTextDetection)
    private lazy var contoursRequest = VNDetectContoursRequest(completionHandler: handleContours)
    
    // YOLO models
    private var yoloRequest: VNCoreMLRequest?
    private var yoloTinyRequest: VNCoreMLRequest?
    
    // Performance tracking
    private var frameCount = 0
    private var fpsTimer: Timer?
    private var lastFrameTime = CACurrentMediaTime()
    
    override init() {
        super.init()
        setupVisionRequests()
        setupYOLOModels()
        startFPSTimer()
    }
    
    // MARK: - Setup Methods
    
    func setupCamera(in view: NSView) {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .high
        
        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else {
            updateStatus("No camera available")
            return
        }
        
        do {
            let videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
            
            if captureSession?.canAddInput(videoInput) == true {
                captureSession?.addInput(videoInput)
            }
            
            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput?.alwaysDiscardsLateVideoFrames = true
            videoDataOutput?.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            
            if captureSession?.canAddOutput(videoDataOutput!) == true {
                captureSession?.addOutput(videoDataOutput!)
            }
            
            videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
            videoPreviewLayer?.videoGravity = .resizeAspectFill
            videoPreviewLayer?.frame = view.bounds
            videoPreviewLayer?.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            
            view.layer?.addSublayer(videoPreviewLayer!)
            
            // Setup overlay layer with autoresizing
            overlayLayer.frame = view.bounds
            overlayLayer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            view.layer?.addSublayer(overlayLayer)
            
            // Create detection layers with autoresizing
            let layerNames = ["face", "hand", "body", "object", "text", "contour"]
            for name in layerNames {
                let layer = CALayer()
                layer.frame = view.bounds
                layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
                detectionLayers[name] = layer
                overlayLayer.addSublayer(layer)
            }
            
            // Store reference to view for coordinate calculations
            self.previewView = view
            
            // Start camera
            DispatchQueue.global(qos: .background).async { [weak self] in
                self?.captureSession?.startRunning()
                DispatchQueue.main.async {
                    self?.isRunning = true
                    self?.updateStatus("Camera running")
                }
            }
            
        } catch {
            updateStatus("Error: \(error.localizedDescription)")
        }
    }
    
    private func setupVisionRequests() {
        // Configure face detection
        faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision3
        faceLandmarksRequest.revision = VNDetectFaceLandmarksRequestRevision3
        
        // Configure hand pose
        handPoseRequest.maximumHandCount = 2
        
        // Configure text detection
        textDetectionRequest.recognitionLevel = .accurate
        textDetectionRequest.usesLanguageCorrection = true
        
        // Configure contours
        contoursRequest.contrastAdjustment = 1.0
        contoursRequest.detectsDarkOnLight = true
    }
    
    private func setupYOLOModels() {
        // Load YOLOv3
        if let modelURL = Bundle.main.url(forResource: "YOLOv3", withExtension: "mlmodelc") {
            do {
                let model = try MLModel(contentsOf: modelURL)
                let visionModel = try VNCoreMLModel(for: model)
                yoloRequest = VNCoreMLRequest(model: visionModel, completionHandler: handleObjectDetection)
                yoloRequest?.imageCropAndScaleOption = .scaleFit
                updateStatus("YOLOv3 loaded")
            } catch {
                print("Error loading YOLOv3: \(error)")
            }
        }
        
        // Load YOLOv3Tiny for faster performance (optional)
        if let modelURL = Bundle.main.url(forResource: "YOLOv3TinyFP16", withExtension: "mlmodelc") {
            do {
                let model = try MLModel(contentsOf: modelURL)
                let visionModel = try VNCoreMLModel(for: model)
                yoloTinyRequest = VNCoreMLRequest(model: visionModel, completionHandler: handleObjectDetection)
                yoloTinyRequest?.imageCropAndScaleOption = .scaleFit
            } catch {
                print("Error loading YOLOv3Tiny: \(error)")
            }
        }
    }
    
    // MARK: - Vision Request Handlers
    
    private func handleFaceDetection(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNFaceObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["face"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            for face in results {
                self.drawBoundingBox(face.boundingBox, in: layer, color: .green, label: "Face")
            }
            
            self.updateDetectionCount()
        }
    }
    
    private func handleFaceLandmarks(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNFaceObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["face"]!
            
            for face in results {
                if let landmarks = face.landmarks {
                    self.drawFaceLandmarks(landmarks, boundingBox: face.boundingBox, in: layer)
                }
            }
        }
    }
    
    private func handleHandPose(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNHumanHandPoseObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["hand"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
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
        guard let results = request.results as? [VNHumanBodyPoseObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["body"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
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
        guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
        
        if !results.isEmpty {
            print("Detected \(results.count) objects")
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["object"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            for object in results where object.confidence > 0.3 {
                let label = object.labels.first?.identifier ?? "Unknown"
                let confidence = object.labels.first?.confidence ?? 0
                let text = "\(label): \(String(format: "%.2f", confidence))"
                
                print("Drawing object: \(text) at \(object.boundingBox)")
                self.drawBoundingBox(object.boundingBox, in: layer, color: .systemBlue, label: text)
            }
            
            self.updateDetectionCount()
        }
    }
    
    private func handleTextDetection(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNRecognizedTextObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["text"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
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
        guard let result = request.results?.first as? VNContoursObservation else { return }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let layer = self.detectionLayers["contour"]!
            layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            let path = CGMutablePath()
            
            for contourIndex in 0..<min(result.contourCount, 10) { // Limit contours for performance
                if let contour = try? result.contour(at: contourIndex) {
                    let points = contour.normalizedPoints
                    
                    for (index, point) in points.enumerated() {
                        // Convert SIMD2<Float> to CGPoint
                        let cgPoint = self.normalizedPointToLayerPoint(CGPoint(x: CGFloat(point.x), y: CGFloat(point.y)))
                        
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
            shapeLayer.lineWidth = 1.0
            layer.addSublayer(shapeLayer)
            
            self.updateDetectionCount()
        }
    }
    
    // MARK: - Drawing Methods
    
    private func drawBoundingBox(_ rect: CGRect, in layer: CALayer, color: NSColor, label: String? = nil) {
        let box = CALayer()
        box.frame = denormalizeRect(rect)
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
            textLayer.frame = CGRect(x: box.frame.minX, y: box.frame.minY - size.height - 2,
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
                // Convert SIMD2<Float> to CGPoint and denormalize
                let normalizedPoint = CGPoint(x: CGFloat(point.x), y: CGFloat(point.y))
                let cgPoint = denormalizePoint(normalizedPoint, in: boundingBox)
                
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
            let cgPoint = normalizedPointToLayerPoint(CGPoint(x: point.location.x, y: point.location.y))
            
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
                path.move(to: normalizedPointToLayerPoint(CGPoint(x: point1.location.x, y: point1.location.y)))
                path.addLine(to: normalizedPointToLayerPoint(CGPoint(x: point2.location.x, y: point2.location.y)))
                
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
            let cgPoint = normalizedPointToLayerPoint(CGPoint(x: point.location.x, y: point.location.y))
            
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
                path.move(to: normalizedPointToLayerPoint(CGPoint(x: point1.location.x, y: point1.location.y)))
                path.addLine(to: normalizedPointToLayerPoint(CGPoint(x: point2.location.x, y: point2.location.y)))
                
                let line = CAShapeLayer()
                line.path = path
                line.strokeColor = NSColor.systemPurple.cgColor
                line.lineWidth = 2.0
                layer.addSublayer(line)
            }
        }
    }
    
    // MARK: - Coordinate Conversion
    
    private func denormalizeRect(_ rect: CGRect) -> CGRect {
        guard let view = previewView else { return .zero }
        let size = view.bounds.size
        // Don't flip Y coordinate for macOS
        return CGRect(
            x: rect.minX * size.width,
            y: rect.minY * size.height,
            width: rect.width * size.width,
            height: rect.height * size.height
        )
    }
    
    private func denormalizePoint(_ point: CGPoint, in rect: CGRect) -> CGPoint {
        guard let view = previewView else { return .zero }
        let size = view.bounds.size
        // Don't flip Y coordinate for macOS
        let x = (rect.minX + point.x * rect.width) * size.width
        let y = (rect.minY + point.y * rect.height) * size.height
        return CGPoint(x: x, y: y)
    }
    
    private func normalizedPointToLayerPoint(_ point: CGPoint) -> CGPoint {
        guard let view = previewView else { return .zero }
        let size = view.bounds.size
        // Don't flip Y coordinate for macOS
        return CGPoint(x: point.x * size.width, y: point.y * size.height)
    }
    
    // MARK: - Control Methods
    
    func toggleCamera() {
        if isRunning {
            captureSession?.stopRunning()
            isRunning = false
            updateStatus("Camera stopped")
        } else {
            DispatchQueue.global(qos: .background).async { [weak self] in
                self?.captureSession?.startRunning()
                DispatchQueue.main.async {
                    self?.isRunning = true
                    self?.updateStatus("Camera running")
                }
            }
        }
    }
    
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
        frameCount += 1
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
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
}

// MARK: - App Entry Point
@main
struct objectdetectionApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1024, height: 768)
    }
}
