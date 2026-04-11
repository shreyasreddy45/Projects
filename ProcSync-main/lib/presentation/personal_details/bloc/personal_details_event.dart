import 'package:equatable/equatable.dart';
import '../models/personal_details_model.dart';

abstract class PersonalDetailsEvent extends Equatable {
  const PersonalDetailsEvent();

  @override
  List<Object?> get props => [];
}

class SubmitPersonalDetails extends PersonalDetailsEvent {
  final PersonalDetailsModel personalDetails;
  final String userId;

  const SubmitPersonalDetails(this.personalDetails, this.userId);

  @override
  List<Object?> get props => [personalDetails, userId];
}
