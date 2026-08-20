import 'package:flutter/material.dart';

class CowSilhouettePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4;

    final body = Rect.fromCenter(
      center: Offset(size.width / 2, size.height / 2),
      width: size.width * 0.65,
      height: size.height * 0.30,
    );

    canvas.drawOval(body, paint);

    // Legs
    canvas.drawLine(
      Offset(size.width * 0.25, size.height * 0.55),
      Offset(size.width * 0.25, size.height * 0.75),
      paint,
    );

    canvas.drawLine(
      Offset(size.width * 0.40, size.height * 0.55),
      Offset(size.width * 0.40, size.height * 0.75),
      paint,
    );

    canvas.drawLine(
      Offset(size.width * 0.60, size.height * 0.55),
      Offset(size.width * 0.60, size.height * 0.75),
      paint,
    );

    canvas.drawLine(
      Offset(size.width * 0.75, size.height * 0.55),
      Offset(size.width * 0.75, size.height * 0.75),
      paint,
    );

    // Head
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(size.width * 0.83, size.height * 0.40),
        width: 70,
        height: 50,
      ),
      paint,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
