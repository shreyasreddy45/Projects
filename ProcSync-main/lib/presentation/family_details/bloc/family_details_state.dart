import 'package:equatable/equatable.dart';

abstract class FamilyDetailsState extends Equatable {
  const FamilyDetailsState();

  @override
  List<Object?> get props => [];
}

class FamilyDetailsInitial extends FamilyDetailsState {}

class FamilyDetailsSubmitting extends FamilyDetailsState {}

class FamilyDetailsSuccess extends FamilyDetailsState {}

class FamilyDetailsFailure extends FamilyDetailsState {
  final String error;

  const FamilyDetailsFailure(this.error);

  @override
  List<Object?> get props => [error];
}
