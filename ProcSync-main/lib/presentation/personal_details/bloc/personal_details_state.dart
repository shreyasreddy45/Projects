import 'package:equatable/equatable.dart';

abstract class PersonalDetailsState extends Equatable {
  const PersonalDetailsState();

  @override
  List<Object?> get props => [];
}

class PersonalDetailsInitial extends PersonalDetailsState {}

class PersonalDetailsSubmitting extends PersonalDetailsState {}

class PersonalDetailsSuccess extends PersonalDetailsState {}

class PersonalDetailsFailure extends PersonalDetailsState {
  final String error;

  const PersonalDetailsFailure(this.error);

  @override
  List<Object?> get props => [error];
}
