import 'package:flutter/material.dart';

import '../../models/scoring_result.dart';
import '../../services/sync_service.dart';
import '../explain/scorecard_screen.dart';

/// The end of a capture — and, when the server can be reached, the beginning
/// of the only part the farmer actually came for.
///
/// This screen used to say "Session saved ✓ / Will sync when online" and stop
/// there. Everything the pipeline produced — the scores, the refusals and
/// their reasons, the weight — landed on the server and was visible only in a
/// browser. The person holding the phone in front of the animal never saw any
/// of it.
///
/// So the capture is now pushed immediately rather than waiting on a
/// connectivity event, and the result is offered as a scorecard. The queue is
/// still there and still owns retries: if the upload fails, the session stays
/// pending exactly as before and this screen says so plainly.
class SessionSavedScreen extends StatefulWidget {
  /// Local SQLite id of the session just written.
  ///
  /// Optional so that any existing caller that pushes this screen without one
  /// still compiles and still behaves the way it always did — saved, queued,
  /// no scoring attempt.
  final String? localId;

  /// Side photograph, passed through to the scorecard so a trait with no
  /// rendered overlay can still show the picture it was measured from.
  final String? sidePhotoPath;

  const SessionSavedScreen({super.key, this.localId, this.sidePhotoPath});

  @override
  State<SessionSavedScreen> createState() => _SessionSavedScreenState();
}

enum _Stage { queued, uploading, scored, failed, unreadable }

class _SessionSavedScreenState extends State<SessionSavedScreen> {
  _Stage _stage = _Stage.queued;
  ScoringResult? _result;

  @override
  void initState() {
    super.initState();
    if (widget.localId != null) {
      _stage = _Stage.uploading;
      _score();
    }
  }

  Future<void> _score() async {
    final id = widget.localId;
    if (id == null) return;

    final json = await SyncService.instance.uploadSessionNow(id);
    if (!mounted) return;

    if (json == null) {
      setState(() => _stage = _Stage.failed);
      return;
    }

    try {
      final parsed = ScoringResult.fromJson(json);
      setState(() {
        _result = parsed;
        _stage = _Stage.scored;
      });
    } catch (e) {
      // The upload itself succeeded and the result is stored, so this is not a
      // lost capture - only a scorecard this build cannot render. Reporting it
      // as a failed upload would send someone to look at the network for a bug
      // that is in the app.
      debugPrint('SessionSavedScreen: could not parse the scorecard: $e');
      setState(() => _stage = _Stage.unreadable);
    }
  }

  void _openScorecard() {
    final result = _result;
    if (result == null) return;
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ScorecardScreen(
          result: result,
          capturedPhotoPath: widget.sidePhotoPath,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Session Saved'),
        automaticallyImplyLeading: false, // No back button
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            children: [
              const Spacer(),
              _icon(),
              const SizedBox(height: 20),
              Text(
                'Session saved ✓',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 10),
              _status(context),
              const Spacer(),
              ..._actions(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _icon() {
    switch (_stage) {
      case _Stage.uploading:
        return const SizedBox(
          width: 84,
          height: 84,
          child: CircularProgressIndicator(strokeWidth: 5),
        );
      case _Stage.scored:
        final demo = _result?.isDemonstration ?? true;
        // A tick over an honest refusal would read as success. Nothing
        // measured is not a failure and not an achievement - it is a result.
        final measuredNothing = (_result?.scoredCount ?? 0) == 0;
        if (demo || measuredNothing) {
          return Icon(
            demo
                ? Icons.warning_amber_rounded
                : Icons.do_not_disturb_on_outlined,
            size: 90,
            color: const Color(0xFFEF6C00),
          );
        }
        return const Icon(
          Icons.verified_outlined,
          size: 90,
          color: Color(0xFF2E7D32),
        );
      case _Stage.failed:
        return const Icon(Icons.cloud_off, size: 90, color: Colors.grey);
      case _Stage.unreadable:
        return const Icon(
          Icons.help_outline,
          size: 90,
          color: Color(0xFFEF6C00),
        );
      case _Stage.queued:
        return const Icon(
          Icons.check_circle_outline,
          color: Colors.green,
          size: 90,
        );
    }
  }

  Widget _status(BuildContext context) {
    final grey = TextStyle(fontSize: 16, color: Colors.grey.shade700);

    switch (_stage) {
      case _Stage.uploading:
        return Column(
          children: [
            Text('Scoring…', style: grey, textAlign: TextAlign.center),
            const SizedBox(height: 6),
            Text(
              'Sending the photographs and measuring the animal. This takes a '
              'few seconds.',
              style: TextStyle(fontSize: 13.5, color: Colors.grey.shade600),
              textAlign: TextAlign.center,
            ),
          ],
        );

      case _Stage.scored:
        final result = _result!;
        if (result.isDemonstration) {
          return Column(
            children: [
              const Text(
                'Scored with DEMONSTRATION DATA',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFFE65100),
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 6),
              Text(
                'Nothing could be measured from these photographs, so the '
                'server answered with placeholders. Do not act on them.',
                style: TextStyle(fontSize: 13.5, color: Colors.grey.shade700),
                textAlign: TextAlign.center,
              ),
            ],
          );
        }
        if (result.scoredCount == 0) {
          return Column(
            children: [
              const Text(
                'Nothing could be measured',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFFE65100),
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 6),
              Text(
                'Every trait was refused. Open the scorecard to see why — most '
                'often the ear tag was not photographed close up.',
                style: TextStyle(fontSize: 13.5, color: Colors.grey.shade700),
                textAlign: TextAlign.center,
              ),
            ],
          );
        }
        return Column(
          children: [
            Text(
              '${result.scoredCount} of ${result.traits.length} traits '
              'measured',
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 6),
            Text(
              result.weightRange == null
                  ? 'The rest were refused, each with a reason.'
                  : 'Weight ${result.weightRange}. The rest were refused, '
                        'each with a reason.',
              style: TextStyle(fontSize: 13.5, color: Colors.grey.shade700),
              textAlign: TextAlign.center,
            ),
          ],
        );

      case _Stage.failed:
        return Column(
          children: [
            Text(
              'Saved on this phone. Not sent yet.',
              style: grey,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 6),
            Text(
              'The server could not be reached. Nothing is lost — the capture '
              'will be sent automatically as soon as there is a connection.',
              style: TextStyle(fontSize: 13.5, color: Colors.grey.shade600),
              textAlign: TextAlign.center,
            ),
          ],
        );

      case _Stage.unreadable:
        return Column(
          children: [
            Text(
              'Sent and stored on the server.',
              style: grey,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 6),
            Text(
              'This version of the app could not display the scorecard that '
              'came back. Nothing is lost — it is on the server, and it is in '
              'this animal’s history.',
              style: TextStyle(fontSize: 13.5, color: Colors.grey.shade600),
              textAlign: TextAlign.center,
            ),
          ],
        );

      case _Stage.queued:
        return Text(
          'Will sync when online.',
          style: grey,
          textAlign: TextAlign.center,
        );
    }
  }

  List<Widget> _actions(BuildContext context) {
    final done = SizedBox(
      width: double.infinity,
      height: 55,
      child: _stage == _Stage.scored
          ? OutlinedButton(
              onPressed: () =>
                  Navigator.popUntil(context, (route) => route.isFirst),
              child: const Text('DONE', style: TextStyle(fontSize: 18)),
            )
          : ElevatedButton(
              onPressed: _stage == _Stage.uploading
                  ? null
                  : () => Navigator.popUntil(context, (route) => route.isFirst),
              child: const Text('DONE', style: TextStyle(fontSize: 18)),
            ),
    );

    return [
      if (_stage == _Stage.scored) ...[
        SizedBox(
          width: double.infinity,
          height: 55,
          child: FilledButton.icon(
            onPressed: _openScorecard,
            icon: const Icon(Icons.assignment_outlined),
            label: const Text(
              'VIEW SCORECARD',
              style: TextStyle(fontSize: 17),
            ),
          ),
        ),
        const SizedBox(height: 12),
      ],
      if (_stage == _Stage.failed) ...[
        SizedBox(
          width: double.infinity,
          height: 55,
          child: FilledButton.icon(
            onPressed: () {
              setState(() => _stage = _Stage.uploading);
              _score();
            },
            icon: const Icon(Icons.refresh),
            label: const Text('TRY AGAIN', style: TextStyle(fontSize: 17)),
          ),
        ),
        const SizedBox(height: 12),
      ],
      done,
    ];
  }
}
