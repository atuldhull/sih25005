import 'package:flutter/material.dart';

import '../../services/api_service.dart';

/// Feature (i): ask anything about ONE animal, answered from that animal's
/// record and nothing else.
///
/// The animal is fixed for the life of the screen. That is a deliberate
/// restriction rather than a missing feature: the server answers strictly from
/// one record, so letting the conversation drift between animals would produce
/// answers about the wrong cow with no visible seam.
class ChatScreen extends StatefulWidget {
  final String animalId;

  /// Something human to put in the title bar - usually the breed. Optional,
  /// because the tag ID alone is a perfectly good identifier.
  final String? animalLabel;

  const ChatScreen({super.key, required this.animalId, this.animalLabel});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final ApiService _api = ApiService();
  final TextEditingController _input = TextEditingController();
  final ScrollController _scroll = ScrollController();
  final List<_Message> _messages = [];

  String _language = 'auto';
  bool _sending = false;

  /// `auto` lets the server detect the language of the question and reply in
  /// kind, which is what a farmer switching between Hindi and English mid
  /// sentence actually needs.
  static const Map<String, String> _languages = {
    'auto': 'Auto',
    'en': 'English',
    'hi': 'हिंदी',
    'kn': 'ಕನ್ನಡ',
  };

  static const List<String> _suggestions = [
    'What is her weight?',
    'वज़न कितना है?',
    'Is she healthy?',
    'What should I feed her?',
    'How much milk does she give?',
  ];

  @override
  void initState() {
    super.initState();
    _messages.add(
      _Message.assistant(
        'Ask me anything about animal ${widget.animalId}. I answer only from '
        'her record — her breed, her past sessions, and standard care advice. '
        'Hindi and Kannada work too.',
        isIntro: true,
      ),
    );
  }

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    _api.close();
    super.dispose();
  }

  Future<void> _send([String? preset]) async {
    final text = (preset ?? _input.text).trim();
    if (text.isEmpty || _sending) return;

    setState(() {
      _messages.add(_Message.user(text));
      _sending = true;
      _input.clear();
    });
    _scrollToEnd();

    final reply = await _api.chat(
      widget.animalId,
      text,
      language: _language,
    );

    if (!mounted) return;

    setState(() {
      _sending = false;
      final error = reply['error'];
      if (error != null) {
        _messages.add(_Message.error(error.toString()));
      } else {
        _messages.add(
          _Message.assistant(
            (reply['answer'] ?? '').toString(),
            disclaimer: reply['disclaimer']?.toString(),
            escalate: reply['escalate'] == true,
            sources: (reply['sources'] as List? ?? [])
                .map((e) => e.toString())
                .toList(),
          ),
        );
      }
    });
    _scrollToEnd();
  }

  void _scrollToEnd() {
    // After the frame, or the list has not been laid out at its new length yet
    // and the jump lands short of the message just added.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final label = widget.animalLabel;

    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Assistant'),
            Text(
              label == null || label.isEmpty
                  ? widget.animalId
                  : '$label · ${widget.animalId}',
              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w400),
            ),
          ],
        ),
        actions: [
          PopupMenuButton<String>(
            tooltip: 'Reply language',
            icon: const Icon(Icons.translate),
            initialValue: _language,
            onSelected: (v) => setState(() => _language = v),
            itemBuilder: (_) => _languages.entries
                .map(
                  (e) => PopupMenuItem<String>(
                    value: e.key,
                    child: Text(e.value),
                  ),
                )
                .toList(),
          ),
        ],
      ),
      body: Column(
        children: [
          if (_language != 'auto')
            Container(
              width: double.infinity,
              color: Colors.grey.shade200,
              padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 16),
              child: Text(
                'Replying in ${_languages[_language]}',
                style: TextStyle(fontSize: 12, color: Colors.grey.shade800),
              ),
            ),

          Expanded(
            child: ListView.builder(
              controller: _scroll,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length + (_sending ? 1 : 0),
              itemBuilder: (context, i) {
                if (i == _messages.length) return const _ThinkingBubble();
                return _Bubble(message: _messages[i]);
              },
            ),
          ),

          if (_messages.length <= 1) _suggestionStrip(),

          SafeArea(
            top: false,
            child: Container(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
              decoration: BoxDecoration(
                border: Border(top: BorderSide(color: Colors.grey.shade300)),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _input,
                      enabled: !_sending,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _send(),
                      minLines: 1,
                      maxLines: 4,
                      decoration: const InputDecoration(
                        hintText: 'Ask about this animal…',
                        border: OutlineInputBorder(),
                        isDense: true,
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 12,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    onPressed: _sending ? null : () => _send(),
                    icon: const Icon(Icons.send),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _suggestionStrip() {
    return SizedBox(
      height: 44,
      child: ListView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        children: [
          for (final s in _suggestions)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: ActionChip(label: Text(s), onPressed: () => _send(s)),
            ),
        ],
      ),
    );
  }
}

enum _Role { user, assistant, error }

class _Message {
  final _Role role;
  final String text;
  final String? disclaimer;
  final bool escalate;
  final List<String> sources;
  final bool isIntro;

  const _Message._(
    this.role,
    this.text, {
    this.disclaimer,
    this.escalate = false,
    this.sources = const [],
    this.isIntro = false,
  });

  factory _Message.user(String t) => _Message._(_Role.user, t);

  factory _Message.assistant(
    String t, {
    String? disclaimer,
    bool escalate = false,
    List<String> sources = const [],
    bool isIntro = false,
  }) => _Message._(
    _Role.assistant,
    t,
    disclaimer: disclaimer,
    escalate: escalate,
    sources: sources,
    isIntro: isIntro,
  );

  factory _Message.error(String t) => _Message._(_Role.error, t);
}

class _Bubble extends StatelessWidget {
  final _Message message;

  const _Bubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == _Role.user;
    final isError = message.role == _Role.error;

    final Color background;
    if (isUser) {
      background = Theme.of(context).colorScheme.primaryContainer;
    } else if (isError) {
      background = const Color(0xFFFDECEA);
    } else {
      background = Colors.grey.shade100;
    }

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.82,
        ),
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: background,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isError ? const Color(0xFFD32F2F) : Colors.transparent,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (isError)
              Row(
                children: [
                  const Icon(
                    Icons.error_outline,
                    size: 16,
                    color: Color(0xFFB71C1C),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    'Could not answer',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Colors.red.shade900,
                    ),
                  ),
                ],
              ),
            if (isError) const SizedBox(height: 4),

            Text(
              message.text,
              style: TextStyle(
                height: 1.4,
                color: isError ? Colors.red.shade900 : null,
                fontStyle: message.isIntro ? FontStyle.italic : null,
              ),
            ),

            // An escalation is the assistant saying this needs a person, and
            // it must not read like the rest of the paragraph.
            if (message.escalate) ...[
              const SizedBox(height: 10),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF3E0),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color(0xFFEF6C00)),
                ),
                child: const Row(
                  children: [
                    Icon(
                      Icons.local_hospital_outlined,
                      size: 16,
                      color: Color(0xFFE65100),
                    ),
                    SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        'Contact your veterinary officer about this.',
                        style: TextStyle(
                          fontSize: 12.5,
                          color: Color(0xFFE65100),
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],

            if (message.sources.isNotEmpty) ...[
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                runSpacing: 4,
                children: message.sources
                    .map(
                      (s) => Chip(
                        label: Text(s, style: const TextStyle(fontSize: 11)),
                        visualDensity: VisualDensity.compact,
                        materialTapTargetSize:
                            MaterialTapTargetSize.shrinkWrap,
                        padding: EdgeInsets.zero,
                      ),
                    )
                    .toList(),
              ),
            ],

            if (message.disclaimer != null &&
                message.disclaimer!.trim().isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                message.disclaimer!,
                style: TextStyle(
                  fontSize: 11.5,
                  height: 1.3,
                  color: Colors.grey.shade600,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ThinkingBubble extends StatelessWidget {
  const _ThinkingBubble();

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.grey.shade100,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 10),
            Text(
              'Thinking…',
              style: TextStyle(color: Colors.grey.shade700),
            ),
          ],
        ),
      ),
    );
  }
}
