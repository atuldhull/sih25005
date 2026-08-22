import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pashu_scorer/models/scoring_result.dart';

/// The decode has to survive a real server response, byte for byte.
///
/// This exists because the app once reported a capture as failed to upload
/// when the upload had in fact succeeded and been stored: the failure was in
/// parsing what came back, and the screen could not tell the two apart. A
/// fixture recorded from the live server is the only thing that would have
/// caught it.
void main() {
  test('parses a real ml-pipeline response from the server', () {
    final raw = File('test/_stored_result.json').readAsStringSync();
    final json = jsonDecode(raw) as Map<String, dynamic>;

    final result = ScoringResult.fromJson(json);

    expect(result.engine, 'ml-pipeline');
    expect(result.isDemonstration, isFalse);
    expect(result.animalId, '356279812346');
    expect(result.traits.length, 20);
    expect(result.scoredCount, greaterThan(0));
    expect(result.weightRange, isNotNull);
    expect(result.weightCrossCheck, contains('DISAGREES'));
    expect(result.predictedGroup, 'red_zebu');
    expect(result.breedVerifyStatus, 'unverified');

    // A refused trait must carry its reason, or the scorecard shows a blank
    // where the most useful sentence on the screen belongs.
    final refused = result.traits.where((t) => !t.isScored);
    expect(refused, isNotEmpty);
    expect(refused.every((t) => t.notScoredReason != null), isTrue);
  });

  test('survives a response with every optional field missing', () {
    final result = ScoringResult.fromJson({});
    expect(result.isDemonstration, isTrue);
    expect(result.traits, isEmpty);
    expect(result.weightRange, isNull);
  });
}
