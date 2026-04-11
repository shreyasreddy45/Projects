import 'dart:async';
import 'dart:io';

import 'package:bloc/bloc.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:meta/meta.dart';

part 'upload_event.dart';
part 'upload_state.dart';

class UploadBloc extends Bloc<UploadEvent, UploadState> {
  UploadBloc() : super(UploadInitial()) {
    on<UploadFileEvent>(_onUploadFile);
    on<LoadUploadsEvent>(_onLoadUploads);
  }

  Future<void> _onUploadFile(
      UploadFileEvent event, Emitter<UploadState> emit) async {
    emit(UploadLoading());

    try {
      final file = File(event.filePath);

      if (!file.existsSync()) {
        emit(UploadError('Selected file does not exist.'));
        return;
      }

      final extension = event.filePath.split('.').last.toLowerCase();
      final allowedExtensions = ['jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx'];

      if (!allowedExtensions.contains(extension)) {
        emit(UploadError('File type not supported.'));
        return;
      }

      final fileName =
          '${DateTime.now().millisecondsSinceEpoch}_${file.uri.pathSegments.last}';
      final ref = FirebaseStorage.instance.ref().child('uploads/$fileName');

      final metadata =
          SettableMetadata(contentType: _getContentType(extension));

      final uploadTask = ref.putFile(file, metadata);

      final snapshot = await uploadTask.whenComplete(() {});
      if (snapshot.state == TaskState.success) {
        final url = await ref.getDownloadURL();

        await FirebaseFirestore.instance.collection('uploads').add({
          'uploaderId': event.userId,
          'uploaderName': event.userName,
          'title': event.title,
          'fileUrl': url,
          'timestamp': Timestamp.now(),
        });

        emit(UploadSuccess());
        add(LoadUploadsEvent(event.userId, event.role));
      } else {
        emit(UploadError('Upload failed: task state was not success'));
      }
    } catch (e) {
      emit(UploadError('Upload failed: ${e.toString()}'));
    }
  }

  Future<void> _onLoadUploads(
      LoadUploadsEvent event, Emitter<UploadState> emit) async {
    emit(UploadLoading());

    try {
      QuerySnapshot snapshot;
      if (event.role == 'student') {
        snapshot = await FirebaseFirestore.instance
            .collection('uploads')
            .where('uploaderId', isEqualTo: event.userId)
            .orderBy('timestamp', descending: true)
            .get();
      } else {
        snapshot = await FirebaseFirestore.instance
            .collection('uploads')
            .orderBy('timestamp', descending: true)
            .get();
      }

      final uploads = snapshot.docs
          .map((doc) => doc.data() as Map<String, dynamic>)
          .toList();
      emit(UploadLoaded(uploads));
    } catch (e) {
      emit(UploadError('Failed to load uploads: ${e.toString()}'));
    }
  }

  String _getContentType(String extension) {
    switch (extension) {
      case 'jpg':
      case 'jpeg':
        return 'image/jpeg';
      case 'png':
        return 'image/png';
      case 'pdf':
        return 'application/pdf';
      case 'doc':
        return 'application/msword';
      case 'docx':
        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
      default:
        return 'application/octet-stream';
    }
  }
}
