import 'package:equatable/equatable.dart';
import '../models/student_info_model.dart';

abstract class StudentInfoState extends Equatable {
  @override
  List<Object?> get props => [];
}

class StudentInfoInitial extends StudentInfoState {}

class StudentInfoLoading extends StudentInfoState {}

class StudentInfoLoaded extends StudentInfoState {
  final StudentInfo studentInfo;
  StudentInfoLoaded(this.studentInfo);

  @override
  List<Object?> get props => [studentInfo];
}

class StudentInfoError extends StudentInfoState {
  final String message;
  StudentInfoError(this.message);

  @override
  List<Object?> get props => [message];
}
