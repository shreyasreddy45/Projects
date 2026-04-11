import 'dart:io';
import 'package:equatable/equatable.dart';

abstract class StudentInfoEvent extends Equatable {
  @override
  List<Object?> get props => [];
}

class LoadStudentInfo extends StudentInfoEvent {
  final String uid;
  LoadStudentInfo(this.uid);

  @override
  List<Object?> get props => [uid];
}

class UploadProfilePhoto extends StudentInfoEvent {
  final String uid;
  final File imageFile;
  UploadProfilePhoto({required this.uid, required this.imageFile});

  @override
  List<Object?> get props => [uid, imageFile];
}

class UpdatePersonalDetails extends StudentInfoEvent {
  final String uid;
  final Map<String, dynamic> personalDetails;
  UpdatePersonalDetails({required this.uid, required this.personalDetails});

  @override
  List<Object?> get props => [uid, personalDetails];
}

class UpdateParentDetails extends StudentInfoEvent {
  final String uid;
  final Map<String, dynamic> parentDetails;
  UpdateParentDetails({required this.uid, required this.parentDetails});

  @override
  List<Object?> get props => [uid, parentDetails];
}

class UpdateEducationDetails extends StudentInfoEvent {
  final String uid;
  final Map<String, dynamic> educationDetails;
  UpdateEducationDetails({required this.uid, required this.educationDetails});

  @override
  List<Object?> get props => [uid, educationDetails];
}
