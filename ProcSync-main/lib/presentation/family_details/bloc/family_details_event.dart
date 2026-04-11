import 'package:equatable/equatable.dart';
import '../models/family_details_model.dart';

abstract class FamilyDetailsEvent extends Equatable {
  const FamilyDetailsEvent();

  @override
  List<Object?> get props => [];
}

class SubmitFamilyDetails extends FamilyDetailsEvent {
  final FamilyDetailsModel details;
  final String userId;

  const SubmitFamilyDetails(this.details, this.userId);

  @override
  List<Object?> get props => [details, userId];
}
