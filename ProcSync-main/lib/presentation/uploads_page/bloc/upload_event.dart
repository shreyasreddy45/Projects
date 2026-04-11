part of 'upload_bloc.dart';

@immutable
abstract class UploadEvent {}

class UploadFileEvent extends UploadEvent {
  final String filePath;
  final String title;
  final String userId;
  final String userName;
  final String role;

  UploadFileEvent(
      this.filePath, this.title, this.userId, this.userName, this.role);
}

class LoadUploadsEvent extends UploadEvent {
  final String userId;
  final String role;

  LoadUploadsEvent(this.userId, this.role);
}
