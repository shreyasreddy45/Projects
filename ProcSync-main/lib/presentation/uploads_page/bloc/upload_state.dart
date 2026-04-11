part of 'upload_bloc.dart';

@immutable
abstract class UploadState {}

class UploadInitial extends UploadState {}

class UploadLoading extends UploadState {}

class UploadSuccess extends UploadState {}

class UploadLoaded extends UploadState {
  final List<Map<String, dynamic>> uploads;

  UploadLoaded(this.uploads);
}

class UploadError extends UploadState {
  final String message;

  UploadError(this.message);
}
