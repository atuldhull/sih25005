class CaptureSession {
  String? tagId;
  String? sidePhotoPath;
  String? rearPhotoPath;
  String? videoPath;

  /// Close-up photograph of the ear tag, captured on ScanTagScreen.
  ///
  /// Everything the server measures in centimetres depends on this. In the
  /// side photo the tag is a thumbnail and the detector frequently misses it
  /// entirely; in a close-up it fills the frame, needs no detector, and its
  /// printed 18 mm digit row gives a scale directly. The server then carries
  /// that scale to the side photo using the tag itself as a bridge.
  ///
  /// Optional: without it the angle traits still score and the class-C traits
  /// refuse honestly.
  String? tagPhotoPath;

  CaptureSession({
    this.tagId,
    this.sidePhotoPath,
    this.rearPhotoPath,
    this.videoPath,
    this.tagPhotoPath,
  });
}
