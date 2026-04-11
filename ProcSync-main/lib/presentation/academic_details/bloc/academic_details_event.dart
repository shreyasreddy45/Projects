import 'package:equatable/equatable.dart';
import '../models/academic_course_model.dart';

abstract class AcademicDetailsEvent extends Equatable {
  @override
  List<Object> get props => [];
}

class SubmitAcademicDetails extends AcademicDetailsEvent {
  final String userId;
  final List<AcademicCourseModel> courses;

  SubmitAcademicDetails({required this.userId, required this.courses});

  @override
  List<Object> get props => [userId, courses];
}
