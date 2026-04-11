import 'package:equatable/equatable.dart';

abstract class AcademicDetailsState extends Equatable {
  @override
  List<Object> get props => [];
}

class AcademicDetailsInitial extends AcademicDetailsState {}

class AcademicDetailsSubmitting extends AcademicDetailsState {}

class AcademicDetailsSuccess extends AcademicDetailsState {}

class AcademicDetailsFailure extends AcademicDetailsState {
  final String error;
  AcademicDetailsFailure(this.error);
  @override
  List<Object> get props => [error];
}
