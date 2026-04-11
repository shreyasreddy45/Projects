import 'package:cloud_firestore/cloud_firestore.dart';

class UploadModel {
  final String id;
  final String uploaderId;
  final String uploaderName;
  final String title;
  final String fileUrl;
  final DateTime timestamp;

  UploadModel({
    required this.id,
    required this.uploaderId,
    required this.uploaderName,
    required this.title,
    required this.fileUrl,
    required this.timestamp,
  });

  factory UploadModel.fromFirestore(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>;
    return UploadModel(
      id: doc.id,
      uploaderId: data['uploaderId'],
      uploaderName: data['uploaderName'],
      title: data['title'],
      fileUrl: data['fileUrl'],
      timestamp: (data['timestamp'] as Timestamp).toDate(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'uploaderId': uploaderId,
      'uploaderName': uploaderName,
      'title': title,
      'fileUrl': fileUrl,
      'timestamp': Timestamp.fromDate(timestamp),
    };
  }
}
