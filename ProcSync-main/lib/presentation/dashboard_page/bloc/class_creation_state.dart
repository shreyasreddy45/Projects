part of 'class_creation_bloc.dart';

abstract class ClassCreationState {}

class ClassCreationInitial extends ClassCreationState {}

class ClassCreationLoading extends ClassCreationState {}

class ClassCreationSuccess extends ClassCreationState {
  final ClassModel classModel;
  ClassCreationSuccess(this.classModel);
}

class ClassCreationFailure extends ClassCreationState {
  final String error;
  ClassCreationFailure(this.error);
}
