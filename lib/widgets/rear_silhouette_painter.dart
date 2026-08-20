import 'package:flutter/material.dart';

class RearSilhouettePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4;

    // Rump/body shape
    final body = Rect.fromCenter(
      center: Offset(size.width / 2, size.height * 0.4),
      width: size.width * 0.5,
      height: size.height * 0.4,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(body, const Radius.circular(20)),
      paint,
    );

    // Legs
    canvas.drawLine(
      Offset(size.width * 0.35, size.height * 0.6),
      Offset(size.width * 0.35, size.height * 0.8),
      paint,
    );
    canvas.drawLine(
      Offset(size.width * 0.65, size.height * 0.6),
      Offset(size.width * 0.65, size.height * 0.8),
      paint,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
