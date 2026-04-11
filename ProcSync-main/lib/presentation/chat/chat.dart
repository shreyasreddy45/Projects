import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:file_picker/file_picker.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:intl/intl.dart';
import 'package:url_launcher/url_launcher_string.dart';

import 'bloc/ui_bloc.dart';
import 'bloc/ui_event.dart';
import 'bloc/ui_state.dart';
import 'models/user_model.dart';
import '../dashboard_page/models/class_model.dart';
import 'package:ProcSync/routes/app_routes.dart'; // Make sure this file exists with route names

class ChatPage extends StatefulWidget {
  final UserModel currentUser;
  final ClassModel classModel;

  const ChatPage({
    Key? key,
    required this.currentUser,
    required this.classModel,
  }) : super(key: key);

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    context.read<UiBloc>().add(SetUserRoleEvent(widget.currentUser.role));
  }

  void _sendMessage() async {
    final message = _messageController.text.trim();
    if (message.isEmpty) return;

    final currentUser = FirebaseAuth.instance.currentUser;
    if (currentUser == null) return;

    await FirebaseFirestore.instance
        .collection('classrooms')
        .doc(widget.classModel.code)
        .collection('messages')
        .add({
      'text': message,
      'senderId': currentUser.uid,
      'senderName': widget.currentUser.name,
      'senderPhotoUrl': currentUser.photoURL,
      'timestamp': FieldValue.serverTimestamp(),
    });

    _messageController.clear();
    _scrollToBottom();
  }

  void _attachDocument() async {
    final result = await FilePicker.platform.pickFiles();
    if (result != null && result.files.single.path != null) {
      final file = File(result.files.single.path!);
      final fileName = result.files.single.name;

      final ref = FirebaseStorage.instance
          .ref('class_files/${widget.classModel.code}/$fileName');

      await ref.putFile(file);
      final url = await ref.getDownloadURL();

      await FirebaseFirestore.instance
          .collection('classrooms')
          .doc(widget.classModel.code)
          .collection('messages')
          .add({
        'text': '📎 $fileName',
        'fileUrl': url,
        'senderId': FirebaseAuth.instance.currentUser!.uid,
        'senderName': widget.currentUser.name,
        'senderPhotoUrl': FirebaseAuth.instance.currentUser?.photoURL,
        'timestamp': FieldValue.serverTimestamp(),
      });

      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 300), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Widget _buildAvatar(String? photoUrl, String nameInitial, bool isTeacher) {
    return CircleAvatar(
      radius: isTeacher ? 12 : 16,
      backgroundImage: (photoUrl != null && photoUrl.isNotEmpty)
          ? NetworkImage(photoUrl)
          : null,
      child: (photoUrl == null || photoUrl.isEmpty)
          ? Text(nameInitial, style: const TextStyle(fontSize: 12))
          : null,
    );
  }

  Widget _buildMessage(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>? ?? {};
    final isMe = data['senderId'] == FirebaseAuth.instance.currentUser?.uid;
    final String text = data['text'] ?? '';
    final String senderName = data['senderName'] ?? 'Unknown';
    final String senderPhotoUrl = data['senderPhotoUrl'] ?? '';
    final Timestamp? timestamp = data['timestamp'];

    final messageContent = Column(
      crossAxisAlignment:
          isMe ? CrossAxisAlignment.end : CrossAxisAlignment.start,
      children: [
        Text(
          senderName,
          style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w500),
        ),
        const SizedBox(height: 4),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            color: isMe ? Colors.blue[100] : Colors.grey[200],
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            crossAxisAlignment:
                isMe ? CrossAxisAlignment.end : CrossAxisAlignment.start,
            children: [
              if (data['fileUrl'] != null)
                GestureDetector(
                  onTap: () => launchUrlString(data['fileUrl']),
                  child: Text(
                    text,
                    style: const TextStyle(
                      color: Colors.blue,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                )
              else
                Text(text),
              if (timestamp != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    DateFormat('hh:mm a').format(timestamp.toDate()),
                    style: const TextStyle(fontSize: 10, color: Colors.grey),
                  ),
                ),
            ],
          ),
        ),
      ],
    );

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 10),
      child: Row(
        mainAxisAlignment:
            isMe ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: isMe
            ? [
                Flexible(child: messageContent),
                const SizedBox(width: 6),
                _buildAvatar(senderPhotoUrl, senderName[0], false),
              ]
            : [
                _buildAvatar(senderPhotoUrl, senderName[0], false),
                const SizedBox(width: 6),
                Flexible(child: messageContent),
              ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<UiBloc, UiState>(
      builder: (context, state) {
        return Scaffold(
          drawer: Drawer(
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.only(top: 40, bottom: 12),
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey,
                        blurRadius: 2,
                        offset: Offset(0, 1),
                      )
                    ],
                  ),
                  child: ShaderMask(
                    shaderCallback: (Rect bounds) {
                      return const LinearGradient(
                        colors: [
                          Colors.deepPurple,
                          Colors.blue,
                          Colors.lightBlueAccent
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ).createShader(bounds);
                    },
                    child: const Center(
                      child: Text(
                        "ProcSync",
                        style: TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                ),
                const Divider(),
                ListTile(
                  leading: const Icon(Icons.chat_bubble),
                  title: const Text("Stream"),
                  selected: true,
                  selectedTileColor: Colors.grey[300],
                ),
                if (state.role.toLowerCase() == 'student') ...[
                  ListTile(
                    leading: const Icon(Icons.upload),
                    title: const Text("Uploads"),
                    onTap: () {
                      Navigator.pushNamed(
                        context,
                        AppRoutes.uploadsPage,
                        arguments: {'user': widget.currentUser},
                      );
                    },
                  ),
                  ListTile(
                    leading: const Icon(Icons.assignment_ind),
                    title: const Text("Student Info"),
                    onTap: () {
                      Navigator.pushNamed(
                        context,
                        AppRoutes.studentInfoPage,
                        arguments: {
                          'user': widget.currentUser,
                          'classCode': widget.classModel.code,
                        },
                      );
                    },
                  ),
                ] else if (state.role.toLowerCase() == 'teacher') ...[
                  ListTile(
                    leading: const Icon(Icons.upload_file),
                    title: const Text("Docs"),
                    onTap: () {
                      Navigator.pushNamed(
                        context,
                        AppRoutes.uploadsPage,
                        arguments: {'user': widget.currentUser},
                      );
                    },
                  ),
                  ListTile(
                    leading: const Icon(Icons.people),
                    title: const Text("People"),
                    onTap: () {
                      Navigator.pushNamed(
                        context,
                        AppRoutes.peoplePage,
                        arguments: {
                          'currentUser': widget
                              .currentUser, // Pass the logged-in teacher UserModel
                          'classModel': widget
                              .classModel, // ClassModel of the current class
                        },
                      );
                    },
                  ),
                ],
              ],
            ),
          ),
          appBar: AppBar(
            backgroundColor: Colors.white,
            elevation: 0,
            leading: Builder(
              builder: (context) => IconButton(
                icon: const Icon(Icons.menu, color: Colors.black),
                onPressed: () => Scaffold.of(context).openDrawer(),
              ),
            ),
            actions: [
              IconButton(
                icon: const Icon(Icons.more_vert, color: Colors.black),
                onPressed: () {
                  context.read<UiBloc>().add(ShowFeedbackEvent());
                },
              ),
            ],
            title: Text(
              widget.classModel.name,
              style: const TextStyle(color: Colors.black),
            ),
            centerTitle: true,
          ),
          body: Column(
            children: [
              Expanded(
                child: StreamBuilder<QuerySnapshot>(
                  stream: FirebaseFirestore.instance
                      .collection('classrooms')
                      .doc(widget.classModel.code)
                      .collection('messages')
                      .orderBy('timestamp', descending: false)
                      .snapshots(),
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const Center(child: CircularProgressIndicator());
                    }
                    final messages = snapshot.data?.docs ?? [];
                    return ListView(
                      controller: _scrollController,
                      padding: const EdgeInsets.all(8),
                      children: messages
                          .map((doc) => KeyedSubtree(
                                key: ValueKey(doc.id),
                                child: _buildMessage(doc),
                              ))
                          .toList(),
                    );
                  },
                ),
              ),
              if (state.showFeedback)
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Column(
                    children: [
                      TextField(
                        decoration: const InputDecoration(
                          hintText: "Send Feedback",
                          border: OutlineInputBorder(),
                        ),
                        maxLines: 3,
                      ),
                      TextButton(
                        onPressed: () {
                          context.read<UiBloc>().add(CloseFeedbackEvent());
                        },
                        child: const Text("Close"),
                      ),
                    ],
                  ),
                ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.attach_file),
                      onPressed: _attachDocument,
                    ),
                    Expanded(
                      child: TextField(
                        controller: _messageController,
                        decoration: const InputDecoration(
                          hintText: "Message",
                          filled: true,
                          fillColor: Colors.white,
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.all(Radius.circular(30)),
                            borderSide: BorderSide.none,
                          ),
                        ),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.send),
                      onPressed: _sendMessage,
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
