import 'package:flutter_bloc/flutter_bloc.dart';
import 'family_details_event.dart';
import 'family_details_state.dart';
import '../models/family_details_model.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class FamilyDetailsBloc extends Bloc<FamilyDetailsEvent, FamilyDetailsState> {
  FamilyDetailsBloc() : super(FamilyDetailsInitial()) {
    on<SubmitFamilyDetails>(_onSubmit);
  }

  Future<void> _onSubmit(
    SubmitFamilyDetails event,
    Emitter<FamilyDetailsState> emit,
  ) async {
    emit(FamilyDetailsSubmitting());

    try {
      await FirebaseFirestore.instance
          .collection('students')
          .doc(event.userId)
          .collection('info')
          .doc('family')
          .set(event.details.toMap());

      emit(FamilyDetailsSuccess());
    } catch (e) {
      emit(FamilyDetailsFailure(e.toString()));
    }
  }
}
